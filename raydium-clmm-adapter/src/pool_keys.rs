use anchor_client::{Client, Cluster};
use anyhow::{anyhow, format_err};
use arrayref::array_ref;
use raydium_amm_v3::states::{PoolState, TickArrayBitmapExtension, TickArrayState};
use raydium_client::instructions::{
    amount_with_slippage, get_out_put_amount_and_remaining_accounts, get_transfer_inverse_fee, price_to_sqrt_price_x64,
};
use solana_client::nonblocking::rpc_client::RpcClient;
use solana_sdk::{
    instruction::{AccountMeta, Instruction},
    pubkey::Pubkey,
};
use spl_token_2022::{
    extension::StateWithExtensionsMut,
    state::{Account, Mint},
};
use std::{collections::VecDeque, rc::Rc};

use raydium_client::instructions::utils::{deserialize_anchor_account, get_transfer_fee};
use solana_sdk::signature::Keypair;

#[derive(Debug, PartialEq)]
pub struct PoolKeys {
    pub raydium_v3_program: Pubkey,
    pub amm_config_key: Pubkey,
    pub mint0: Pubkey,
    pub mint1: Pubkey,
    pub pool_id_account: Pubkey,
    pub tickarray_bitmap_extension: Pubkey,
    pub amm_config_index: u16,
}

impl PoolKeys {
    // based on https://github.com/raydium-io/raydium-clmm/blob/d0cb69cc953a17b0fa9977914f281cd1d077067f/client/src/main.rs#L81
    // The list of all pool - https://api.raydium.io/v2/ammV3/ammPools
    pub fn new(
        // CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK for mainnet
        raydium_v3_program: Pubkey,
        mut mint0: Pubkey,
        mut mint1: Pubkey,
        // One from https://api.raydium.io/v2/ammV3/ammConfigs
        amm_config_index: u16,
    ) -> Self {
        let (amm_config_key, _) = Pubkey::find_program_address(
            &[
                raydium_amm_v3::states::AMM_CONFIG_SEED.as_bytes(),
                &amm_config_index.to_be_bytes(),
            ],
            &raydium_v3_program,
        );

        if mint0 > mint1 {
            std::mem::swap(&mut mint0, &mut mint1);
        }

        let (pool_id_account, _) = Pubkey::find_program_address(
            &[
                raydium_amm_v3::states::POOL_SEED.as_bytes(),
                amm_config_key.to_bytes().as_ref(),
                mint0.to_bytes().as_ref(),
                mint1.to_bytes().as_ref(),
            ],
            &raydium_v3_program,
        );

        let (tickarray_bitmap_extension, _) = Pubkey::find_program_address(
            &[
                raydium_amm_v3::states::POOL_TICK_ARRAY_BITMAP_SEED.as_bytes(),
                pool_id_account.to_bytes().as_ref(),
            ],
            &raydium_v3_program,
        );

        Self {
            raydium_v3_program,
            amm_config_key,
            mint0,
            mint1,
            pool_id_account,
            tickarray_bitmap_extension,
            amm_config_index,
        }
    }
}

pub struct SwapV2 {
    pub payer: Pubkey,
    pub input_token: Pubkey,
    pub output_token: Pubkey,
    pub base_in: bool,
    pub amount: u64,
    pub limit_price: Option<f64>,
    pub slippage: f64,
}

impl PoolKeys {
    // based on https://github.com/raydium-io/raydium-clmm/blob/d0cb69cc953a17b0fa9977914f281cd1d077067f/client/src/main.rs#L1724
    pub async fn build_swap_instruction(
        &self,
        params: SwapV2,
        rpc_client: &RpcClient,
    ) -> anyhow::Result<Vec<Instruction>> {
        let SwapV2 {
            payer,
            input_token,
            output_token,
            base_in,
            amount,
            limit_price,
            slippage,
        } = params;
        let load_accounts = vec![
            input_token,
            output_token,
            self.amm_config_key,
            self.pool_id_account,
            self.tickarray_bitmap_extension,
            self.mint0,
            self.mint1,
        ];
        let rsps = rpc_client.get_multiple_accounts(&load_accounts).await?;
        let epoch = rpc_client.get_epoch_info().await?.epoch;

        let [user_input_account, user_output_account, amm_config_account, pool_account, tickarray_bitmap_extension_account, mint0_account, mint1_account] =
            array_ref![rsps, 0, 7];

        let mut user_input_token_data = user_input_account
            .as_ref()
            .ok_or_else(|| anyhow!("can't fetch state: input_token"))?
            .data
            .clone();
        let user_input_state = StateWithExtensionsMut::<Account>::unpack(&mut user_input_token_data)?;
        let mut user_output_token_data = user_output_account
            .as_ref()
            .ok_or_else(|| anyhow!("can't fetch state: output_token"))?
            .data
            .clone();
        let user_output_state = StateWithExtensionsMut::<Account>::unpack(&mut user_output_token_data)?;
        let mut mint0_data = mint0_account
            .as_ref()
            .ok_or_else(|| anyhow!("can't fetch state: mint0"))?
            .data
            .clone();
        let mint0_state = StateWithExtensionsMut::<Mint>::unpack(&mut mint0_data)?;
        let mut mint1_data = mint1_account
            .as_ref()
            .ok_or_else(|| anyhow!("can't fetch state: mint1"))?
            .data
            .clone();
        let mint1_state = StateWithExtensionsMut::<Mint>::unpack(&mut mint1_data)?;
        let amm_config_state = deserialize_anchor_account::<raydium_amm_v3::states::AmmConfig>(
            amm_config_account
                .as_ref()
                .ok_or_else(|| anyhow!("can't fetch state: amm_config"))?,
        )?;
        let pool_state = deserialize_anchor_account::<raydium_amm_v3::states::PoolState>(
            pool_account
                .as_ref()
                .ok_or_else(|| anyhow!("can't fetch state: pool_account"))?,
        )?;
        let tickarray_bitmap_extension = deserialize_anchor_account::<raydium_amm_v3::states::TickArrayBitmapExtension>(
            tickarray_bitmap_extension_account
                .as_ref()
                .ok_or_else(|| anyhow!("can't fetch state: tickarray_bitmap_extension"))?,
        )?;
        let zero_for_one = user_input_state.base.mint == pool_state.token_mint_0
            && user_output_state.base.mint == pool_state.token_mint_1;

        let transfer_fee = if base_in {
            if zero_for_one {
                get_transfer_fee(&mint0_state, epoch, amount)
            } else {
                get_transfer_fee(&mint1_state, epoch, amount)
            }
        } else {
            0
        };
        let amount_specified = amount.checked_sub(transfer_fee).unwrap();
        // load tick_arrays
        let mut tick_arrays = self
            .load_cur_and_next_five_tick_array(&rpc_client, &pool_state, &tickarray_bitmap_extension, zero_for_one)
            .await?;

        let mut sqrt_price_limit_x64 = None;
        if limit_price.is_some() {
            let sqrt_price_x64 = price_to_sqrt_price_x64(
                limit_price.unwrap(),
                pool_state.mint_decimals_0,
                pool_state.mint_decimals_1,
            );
            sqrt_price_limit_x64 = Some(sqrt_price_x64);
        }

        let (mut other_amount_threshold, tick_array_indexs) = get_out_put_amount_and_remaining_accounts(
            amount_specified,
            sqrt_price_limit_x64,
            zero_for_one,
            base_in,
            &amm_config_state,
            &pool_state,
            &tickarray_bitmap_extension,
            &mut tick_arrays,
        )
        .map_err(|err| format_err!("can't calc amounts: {err}"))?;
        println!("amount:{}, other_amount_threshold:{}", amount, other_amount_threshold);
        if base_in {
            // calc mint out amount with slippage
            other_amount_threshold = amount_with_slippage(other_amount_threshold, slippage, false);
        } else {
            // calc max in with slippage
            other_amount_threshold = amount_with_slippage(other_amount_threshold, slippage, true);
            // calc max in with transfer_fee
            let transfer_fee = if zero_for_one {
                get_transfer_inverse_fee(&mint0_state, epoch, other_amount_threshold)
            } else {
                get_transfer_inverse_fee(&mint1_state, epoch, other_amount_threshold)
            };
            other_amount_threshold += transfer_fee;
        }

        let mut remaining_accounts = Vec::new();
        remaining_accounts.push(AccountMeta::new_readonly(self.tickarray_bitmap_extension, false));
        let mut accounts = tick_array_indexs
            .into_iter()
            .map(|index| {
                AccountMeta::new(
                    Pubkey::find_program_address(
                        &[
                            raydium_amm_v3::states::TICK_ARRAY_SEED.as_bytes(),
                            self.pool_id_account.to_bytes().as_ref(),
                            &index.to_be_bytes(),
                        ],
                        &self.raydium_v3_program,
                    )
                    .0,
                    false,
                )
            })
            .collect();
        remaining_accounts.append(&mut accounts);

        swap_v2_instruction(
            self.raydium_v3_program,
            payer,
            pool_state.amm_config,
            self.pool_id_account,
            if zero_for_one {
                pool_state.token_vault_0
            } else {
                pool_state.token_vault_1
            },
            if zero_for_one {
                pool_state.token_vault_1
            } else {
                pool_state.token_vault_0
            },
            pool_state.observation_key,
            input_token,
            output_token,
            if zero_for_one {
                pool_state.token_mint_0
            } else {
                pool_state.token_mint_1
            },
            if zero_for_one {
                pool_state.token_mint_1
            } else {
                pool_state.token_mint_0
            },
            remaining_accounts,
            amount,
            other_amount_threshold,
            sqrt_price_limit_x64,
            base_in,
        )
    }

    pub async fn load_cur_and_next_five_tick_array(
        &self,
        rpc_client: &RpcClient,
        pool_state: &PoolState,
        tickarray_bitmap_extension: &TickArrayBitmapExtension,
        zero_for_one: bool,
    ) -> anyhow::Result<VecDeque<TickArrayState>> {
        let (_, mut current_vaild_tick_array_start_index) =
            pool_state.get_first_initialized_tick_array(&Some(*tickarray_bitmap_extension), zero_for_one)?;
        let mut tick_array_keys = Vec::new();
        tick_array_keys.push(
            Pubkey::find_program_address(
                &[
                    raydium_amm_v3::states::TICK_ARRAY_SEED.as_bytes(),
                    self.pool_id_account.to_bytes().as_ref(),
                    &current_vaild_tick_array_start_index.to_be_bytes(),
                ],
                &self.raydium_v3_program,
            )
            .0,
        );
        let mut max_array_size = 5;
        while max_array_size != 0 {
            let next_tick_array_index = pool_state.next_initialized_tick_array_start_index(
                &Some(*tickarray_bitmap_extension),
                current_vaild_tick_array_start_index,
                zero_for_one,
            )?;
            if next_tick_array_index.is_none() {
                break;
            }
            current_vaild_tick_array_start_index = next_tick_array_index.unwrap();
            tick_array_keys.push(
                Pubkey::find_program_address(
                    &[
                        raydium_amm_v3::states::TICK_ARRAY_SEED.as_bytes(),
                        self.pool_id_account.to_bytes().as_ref(),
                        &current_vaild_tick_array_start_index.to_be_bytes(),
                    ],
                    &self.raydium_v3_program,
                )
                .0,
            );
            max_array_size -= 1;
        }
        let tick_array_rsps = rpc_client.get_multiple_accounts(&tick_array_keys).await?;
        let mut tick_arrays = VecDeque::new();
        for tick_array in tick_array_rsps {
            let tick_array_state = deserialize_anchor_account::<raydium_amm_v3::states::TickArrayState>(
                &tick_array.ok_or_else(|| anyhow!("can't fetch state: tick_array"))?,
            )?;
            tick_arrays.push_back(tick_array_state);
        }
        Ok(tick_arrays)
    }
}

fn swap_v2_instruction(
    raydium_v3_program: Pubkey,
    payer: Pubkey,
    amm_config: Pubkey,
    pool_account_key: Pubkey,
    input_vault: Pubkey,
    output_vault: Pubkey,
    observation_state: Pubkey,
    user_input_token: Pubkey,
    user_out_put_token: Pubkey,
    input_vault_mint: Pubkey,
    output_vault_mint: Pubkey,
    remaining_accounts: Vec<AccountMeta>,
    amount: u64,
    other_amount_threshold: u64,
    sqrt_price_limit_x64: Option<u128>,
    is_base_input: bool,
) -> anyhow::Result<Vec<Instruction>> {
    let fake_payer = Keypair::new();
    // It is OK to create a such client, Anchor RequestBuilder doesn't touch blockchain and doesn't sign transactions
    // in our case.
    let client = Client::new(Cluster::Mainnet, Rc::new(fake_payer));
    let program = client.program(raydium_v3_program)?;
    let instructions = program
        .request()
        .accounts(raydium_amm_v3::accounts::SwapSingleV2 {
            payer,
            amm_config,
            pool_state: pool_account_key,
            input_token_account: user_input_token,
            output_token_account: user_out_put_token,
            input_vault,
            output_vault,
            observation_state,
            token_program: spl_token::id(),
            token_program_2022: spl_token_2022::id(),
            memo_program: spl_memo::id(),
            input_vault_mint,
            output_vault_mint,
        })
        .accounts(remaining_accounts)
        .args(raydium_amm_v3::instruction::SwapV2 {
            amount,
            other_amount_threshold,
            sqrt_price_limit_x64: sqrt_price_limit_x64.unwrap_or(0u128),
            is_base_input,
        })
        .instructions()?;
    Ok(instructions)
}

#[cfg(test)]
mod tests {
    use crate::pool_keys::{PoolKeys, SwapV2};
    use dotenvy::dotenv;
    use solana_client::nonblocking::rpc_client::RpcClient;
    use solana_sdk::{
        compute_budget::ComputeBudgetInstruction, message::Message, pubkey, pubkey::Pubkey, transaction::Transaction,
    };

    const RAYDIUM_V3_PROGRAM_MAINNET: Pubkey = pubkey!("CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK");
    #[test]
    fn should_derive_pool_addresses() {
        let wsol = spl_token::native_mint::id();
        let usdc = pubkey!("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v");
        let amm_config_index = 2;
        let pair = PoolKeys::new(RAYDIUM_V3_PROGRAM_MAINNET, wsol, usdc, amm_config_index);

        // curl https://api.raydium.io/v2/ammV3/ammPools | jq '.data[0]'
        let expected = PoolKeys {
            raydium_v3_program: RAYDIUM_V3_PROGRAM_MAINNET,
            amm_config_key: pubkey!("HfERMT5DRA6C1TAqecrJQFpmkf3wsWTMncqnj3RDg5aw"),
            mint0: wsol,
            mint1: usdc,
            pool_id_account: pubkey!("2QdhepnKRTLjjSqPL1PtKNwqrUkoLee5Gqs8bvZhRdMv"),
            tickarray_bitmap_extension: pubkey!("9z9VTNvaTpJuwjn4LSnjHwZgUR9iGuy59BwXTNbxRF6s"),
            amm_config_index,
        };

        assert_eq!(pair, expected);
    }

    #[tokio::test]
    async fn should_simulate_sol_usdc_swap() -> anyhow::Result<()> {
        dotenv().ok();
        let solana_rpc_url = std::env::var("SOLANA_RPC_URL")?;
        let rpc_client = RpcClient::new(solana_rpc_url);

        let wsol = spl_token::native_mint::id();
        let usdc = pubkey!("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v");
        let amm_config_index = 2;
        let pair = PoolKeys::new(RAYDIUM_V3_PROGRAM_MAINNET, wsol, usdc, amm_config_index);

        let payer = pubkey!("DFZgpRgoJP8c2Phx9rP79yn9rUQkWAU1ksnAH6dY7c8U");
        let input_token =
            spl_associated_token_account::get_associated_token_address_with_program_id(&payer, &wsol, &spl_token::id());
        let output_token =
            spl_associated_token_account::get_associated_token_address_with_program_id(&payer, &usdc, &spl_token::id());
        let amount = 100_000_000;

        let params = SwapV2 {
            payer,
            input_token,
            output_token,
            base_in: true,
            amount,
            limit_price: None,
            slippage: 0.01,
        };

        let mut instructions = vec![ComputeBudgetInstruction::set_compute_unit_limit(1400_000u32)];
        instructions.extend(pair.build_swap_instruction(params, &rpc_client).await?);
        let recent_blockhash = rpc_client.get_latest_blockhash().await?;
        let txn = Transaction::new_unsigned(Message::new_with_blockhash(
            &instructions,
            Some(&payer),
            &recent_blockhash,
        ));

        let response = rpc_client.simulate_transaction(&txn).await?.value;
        println!("{:?}", response);

        Ok(())
    }
}
