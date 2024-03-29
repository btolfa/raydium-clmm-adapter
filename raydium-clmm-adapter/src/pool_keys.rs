use solana_sdk::pubkey::Pubkey;

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

#[cfg(test)]
mod tests {
    use crate::pool_keys::PoolKeys;
    use solana_sdk::{pubkey, pubkey::Pubkey};

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
}
