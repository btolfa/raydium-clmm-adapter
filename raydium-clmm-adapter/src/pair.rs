use crate::pool_keys::PoolKeys;
use anyhow::anyhow;
use arrayref::array_ref;
use raydium_amm_v3::states::{AmmConfig, PoolState, TickArrayBitmapExtension, TickArrayState};
use raydium_client::instructions::{deserialize_anchor_account, get_out_put_amount_and_remaining_accounts};
use solana_client::nonblocking::rpc_client::RpcClient;
use spl_token_2022::{
    extension::{
        transfer_fee::{TransferFeeConfig, MAX_FEE_BASIS_POINTS},
        BaseStateWithExtensions, StateWithExtensions,
    },
    state::Mint,
};
use std::collections::VecDeque;

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
            let transfer_fee_config = if zero_for_one {
                self.mint0_transfer_fee.as_ref()
            } else {
                self.mint1_transfer_fee.as_ref()
            };
            transfer_fee_config
                .map(|fee_config| fee_config.calculate_epoch_fee(self.epoch, input_amount))
                .flatten()
                .unwrap_or_default()
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

#[cfg(test)]
mod tests {
    use crate::{pair::Pair, pool_keys::PoolKeys};
    use anchor_client::anchor_lang::prelude::Pubkey;
    use dotenvy::dotenv;
    use solana_client::nonblocking::rpc_client::RpcClient;
    use solana_sdk::pubkey;

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
}
