def _get_module_by_index(self, model, index):
        """
        Maps the flat index from `all_logits` to the specific MoE module.
        In the new Frequency-Disentangled Architecture:
        index 0: Encoder HF MoE (g_a)
        index 1: Decoder HF MoE (g_s)
        """
        if hasattr(model, "get_moe_modules"):
            moe_modules = model.get_moe_modules()
            if index < len(moe_modules):
                return moe_modules[index]
        return None