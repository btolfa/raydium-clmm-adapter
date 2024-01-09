use crate::pool_keys::PoolKeys;
use anchor_client::{
    anchor_lang::{
        prelude::{AccountMeta, Pubkey},
        solana_program::instruction::Instruction,
    },
    Client, Cluster,
};
use anyhow::anyhow;
use arrayref::array_ref;
use raydium_amm_v3::states::{AmmConfig, PoolState, TickArrayBitmapExtension, TickArrayState};
use raydium_client::instructions::{
    amount_with_slippage, deserialize_anchor_account, get_out_put_amount_and_remaining_accounts,
    price_to_sqrt_price_x64,
};
use solana_client::nonblocking::rpc_client::RpcClient;
use solana_sdk::signature::Keypair;
use spl_token_2022::{
    extension::{
        transfer_fee::{TransferFeeConfig, MAX_FEE_BASIS_POINTS},
        BaseStateWithExtensions, StateWithExtensions,
    },
    state::{Account, Mint},
};
use std::{collections::VecDeque, rc::Rc};

pub struct Pair {
    pool_keys: PoolKeys,

    amm_config_state: AmmConfig,
    pool_state: PoolState,
    tickarray_bitmap_extension: TickArrayBitmapExtension,

    epoch: u64,
    mint0_transfer_fee: Option<TransferFeeConfig>,
    mint1_transfer_fee: Option<TransferFeeConfig>,

    tick_arrays_zero_for_one: VecDeque<TickArrayState>,
    tick_arrays_one_for_zero: VecDeque<TickArrayState>,
}

impl Pair {
    pub async fn fetch(pool_keys: PoolKeys, rpc_client: &RpcClient) -> anyhow::Result<Self> {
        let load_accounts = [
            pool_keys.amm_config_key,
            pool_keys.pool_id_account,
            pool_keys.tickarray_bitmap_extension,
            pool_keys.mint0,
            pool_keys.mint1,
        ];

        let rsps = rpc_client
            .get_multiple_accounts(&load_accounts)
            .await?
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| anyhow!("failed to fetch accounts"))?;
        let epoch = rpc_client.get_epoch_info().await?.epoch;

        let [amm_config_account, pool_account, tickarray_bitmap_extension_account, mint0_account, mint1_account] =
            array_ref![rsps, 0, 5];

        let mint0_transfer_fee = StateWithExtensions::<Mint>::unpack(&mint0_account.data)?
            .get_extension::<TransferFeeConfig>()
            .ok()
            .cloned();
        let mint1_transfer_fee = StateWithExtensions::<Mint>::unpack(&mint1_account.data)?
            .get_extension::<TransferFeeConfig>()
            .ok()
            .cloned();

        let amm_config_state = deserialize_anchor_account::<raydium_amm_v3::states::AmmConfig>(amm_config_account)?;
        let pool_state = deserialize_anchor_account::<raydium_amm_v3::states::PoolState>(pool_account)?;
        let tickarray_bitmap_extension = deserialize_anchor_account::<raydium_amm_v3::states::TickArrayBitmapExtension>(
            tickarray_bitmap_extension_account,
        )?;

        let tick_arrays_zero_for_one = pool_keys
            .load_cur_and_next_five_tick_array(rpc_client, &pool_state, &tickarray_bitmap_extension, true)
            .await?;
        let tick_arrays_one_for_zero = pool_keys
            .load_cur_and_next_five_tick_array(rpc_client, &pool_state, &tickarray_bitmap_extension, false)
            .await?;

        Ok(Self {
            pool_keys,
            amm_config_state,
            pool_state,
            tickarray_bitmap_extension,
            epoch,
            mint0_transfer_fee,
            mint1_transfer_fee,
            tick_arrays_zero_for_one,
            tick_arrays_one_for_zero,
        })
    }

    // without transfer fee
    fn estimate_swap_core(&self, input_amount: u64, zero_for_one: bool, is_base_input: bool) -> anyhow::Result<u64> {
        let mut tick_arrays = if zero_for_one {
            self.tick_arrays_zero_for_one.clone()
        } else {
            self.tick_arrays_one_for_zero.clone()
        };
        let (amount_calculated, _) = get_out_put_amount_and_remaining_accounts(
            input_amount,
            None,
            zero_for_one,
            is_base_input,
            &self.amm_config_state,
            &self.pool_state,
            &self.tickarray_bitmap_extension,
            &mut tick_arrays,
        )
        .map_err(|err| anyhow!("failed to calculate output amount: {}", err))?;

        Ok(amount_calculated)
    }

    pub fn estimate_swap(&self, input_amount: u64, zero_for_one: bool, is_base_input: bool) -> anyhow::Result<u64> {
        let transfer_fee = if is_base_input {
            if zero_for_one {
                get_transfer_fee(self.mint0_transfer_fee.as_ref(), self.epoch, input_amount)
            } else {
                get_transfer_fee(self.mint1_transfer_fee.as_ref(), self.epoch, input_amount)
            }
        } else {
            0
        };

        let amount_specified = input_amount
            .checked_sub(transfer_fee)
            .ok_or_else(|| anyhow!("overflow"))?;
        let mut other_amount = self.estimate_swap_core(amount_specified, zero_for_one, is_base_input)?;

        if !is_base_input {
            let transfer_fee = if zero_for_one {
                get_transfer_inverse_fee(self.mint0_transfer_fee.as_ref(), self.epoch, other_amount)
            } else {
                get_transfer_inverse_fee(self.mint1_transfer_fee.as_ref(), self.epoch, other_amount)
            };
            other_amount += transfer_fee
        };

        Ok(other_amount)
    }
}

fn get_transfer_inverse_fee(transfer_fee_config: Option<&TransferFeeConfig>, epoch: u64, post_fee_amount: u64) -> u64 {
    let fee = if let Some(transfer_fee_config) = transfer_fee_config {
        let transfer_fee = transfer_fee_config.get_epoch_fee(epoch);
        if u16::from(transfer_fee.transfer_fee_basis_points) == MAX_FEE_BASIS_POINTS {
            u64::from(transfer_fee.maximum_fee)
        } else {
            transfer_fee_config
                .calculate_inverse_epoch_fee(epoch, post_fee_amount)
                .unwrap()
        }
    } else {
        0
    };
    fee
}

fn get_transfer_fee(transfer_fee_config: Option<&TransferFeeConfig>, epoch: u64, pre_fee_amount: u64) -> u64 {
    transfer_fee_config
        .and_then(|fee_config| fee_config.calculate_epoch_fee(epoch, pre_fee_amount))
        .unwrap_or_default()
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

impl Pair {
    pub async fn build_swap_instructions(
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

        let rsps = rpc_client
            .get_multiple_accounts(&[input_token, output_token])
            .await?
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| anyhow!("failed to fetch accounts"))?;

        let [user_input_account, user_output_account] = array_ref![rsps, 0, 2];

        let user_input_state = StateWithExtensions::<Account>::unpack(&user_input_account.data)?;
        let user_output_state = StateWithExtensions::<Account>::unpack(&user_output_account.data)?;

        let zero_for_one = user_input_state.base.mint == self.pool_state.token_mint_0
            && user_output_state.base.mint == self.pool_state.token_mint_1;

        let transfer_fee = if base_in {
            if zero_for_one {
                get_transfer_fee(self.mint0_transfer_fee.as_ref(), self.epoch, amount)
            } else {
                get_transfer_fee(self.mint1_transfer_fee.as_ref(), self.epoch, amount)
            }
        } else {
            0
        };

        let amount_specified = amount.checked_sub(transfer_fee).ok_or_else(|| anyhow!("overflow"))?;
        let mut tick_arrays = if zero_for_one {
            self.tick_arrays_zero_for_one.clone()
        } else {
            self.tick_arrays_one_for_zero.clone()
        };

        let sqrt_price_limit_x64 = limit_price.map(|limit_price| {
            price_to_sqrt_price_x64(
                limit_price,
                self.pool_state.mint_decimals_0,
                self.pool_state.mint_decimals_1,
            )
        });

        let (mut other_amount_threshold, tick_array_indexs) = get_out_put_amount_and_remaining_accounts(
            amount_specified,
            sqrt_price_limit_x64,
            zero_for_one,
            base_in,
            &self.amm_config_state,
            &self.pool_state,
            &self.tickarray_bitmap_extension,
            &mut tick_arrays,
        )
        .map_err(|err| anyhow!("can't calc amounts: {err}"))?;

        if base_in {
            // calc mint out amount with slippage
            other_amount_threshold = amount_with_slippage(other_amount_threshold, slippage, false);
        } else {
            // calc max in with slippage
            other_amount_threshold = amount_with_slippage(other_amount_threshold, slippage, true);
            // calc max in with transfer_fee
            let transfer_fee = if zero_for_one {
                get_transfer_inverse_fee(self.mint0_transfer_fee.as_ref(), self.epoch, other_amount_threshold)
            } else {
                get_transfer_inverse_fee(self.mint1_transfer_fee.as_ref(), self.epoch, other_amount_threshold)
            };
            other_amount_threshold += transfer_fee;
        }

        let remaining_accounts: Vec<_> = [AccountMeta::new_readonly(
            self.pool_keys.tickarray_bitmap_extension,
            false,
        )]
        .into_iter()
        .chain(tick_array_indexs.into_iter().map(|index| {
            AccountMeta::new(
                Pubkey::find_program_address(
                    &[
                        raydium_amm_v3::states::TICK_ARRAY_SEED.as_bytes(),
                        self.pool_keys.pool_id_account.to_bytes().as_ref(),
                        &index.to_be_bytes(),
                    ],
                    &self.pool_keys.raydium_v3_program,
                )
                .0,
                false,
            )
        }))
        .collect();

        swap_v2_instruction(
            self.pool_keys.raydium_v3_program,
            payer,
            self.pool_state.amm_config,
            self.pool_keys.pool_id_account,
            if zero_for_one {
                self.pool_state.token_vault_0
            } else {
                self.pool_state.token_vault_1
            },
            if zero_for_one {
                self.pool_state.token_vault_1
            } else {
                self.pool_state.token_vault_0
            },
            self.pool_state.observation_key,
            input_token,
            output_token,
            if zero_for_one {
                self.pool_state.token_mint_0
            } else {
                self.pool_state.token_mint_1
            },
            if zero_for_one {
                self.pool_state.token_mint_1
            } else {
                self.pool_state.token_mint_0
            },
            remaining_accounts,
            amount,
            other_amount_threshold,
            sqrt_price_limit_x64,
            base_in,
        )
    }
}

#[allow(clippy::too_many_arguments)]
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
    use crate::{
        pair::{Pair, SwapV2},
        pool_keys::PoolKeys,
    };
    use anchor_client::anchor_lang::{prelude::Pubkey, solana_program::message::Message};
    use dotenvy::dotenv;
    use solana_client::nonblocking::rpc_client::RpcClient;
    use solana_sdk::{compute_budget::ComputeBudgetInstruction, pubkey, transaction::Transaction};

    const RAYDIUM_V3_PROGRAM_MAINNET: Pubkey = pubkey!("CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK");
    #[tokio::test]
    async fn should_estimate_sol_usdc_swap() -> anyhow::Result<()> {
        dotenv().ok();
        let solana_rpc_url = std::env::var("SOLANA_RPC_URL")?;
        let rpc_client = RpcClient::new(solana_rpc_url);

        let wsol = spl_token::native_mint::id();
        let usdc = pubkey!("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v");
        let amm_config_index = 2;
        let pool_keys = PoolKeys::new(RAYDIUM_V3_PROGRAM_MAINNET, wsol, usdc, amm_config_index);

        let pair = Pair::fetch(pool_keys, &rpc_client).await?;
        let other_amount = pair.estimate_swap(1_000_000_000, true, true)?;

        println!("swap 1 SOL to {} amount USDC", other_amount);

        Ok(())
    }

    #[tokio::test]
    async fn should_simulate_sol_usdc_swap() -> anyhow::Result<()> {
        dotenv().ok();
        let solana_rpc_url = std::env::var("SOLANA_RPC_URL")?;
        let rpc_client = RpcClient::new(solana_rpc_url);

        let wsol = spl_token::native_mint::id();
        let usdc = pubkey!("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v");
        let amm_config_index = 2;
        let pool_keys = PoolKeys::new(RAYDIUM_V3_PROGRAM_MAINNET, wsol, usdc, amm_config_index);
        let pair = Pair::fetch(pool_keys, &rpc_client).await?;

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
        instructions.extend(pair.build_swap_instructions(params, &rpc_client).await?);
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
