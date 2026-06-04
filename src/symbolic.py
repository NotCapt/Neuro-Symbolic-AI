"""
Symbolic Conservation Layer module.
"""

import torch
import torch.nn as nn

class SymbolicConservationLayer(nn.Module):
    """Derives GDM and Total from primary predictions in original (non-log) space.
    The additive conservation law is encoded here, not in the loss.
    CSR for P1 and P2 is 100% by construction.
    """
    def forward(self, green_log, dead_log, clover_log):
        # Back-transform primary predictions to original space
        green  = torch.expm1(green_log.clamp(min=0))    # [B]
        dead   = torch.expm1(dead_log.clamp(min=0))     # [B]
        clover = torch.expm1(clover_log.clamp(min=0))   # [B]

        # --- Biological Conservation Laws (exact by construction) ------------
        gdm   = green + clover            # Rule: GDM = Green + Clover
        total = gdm + dead                # Rule: Total = GDM + Dead

        return {
            # Log-space (for regression loss comparison with log-transformed targets)
            'Dry_Green_g':  green_log,
            'Dry_Dead_g':   dead_log,
            'Dry_Clover_g': clover_log,
            'GDM_g':        torch.log1p(gdm.clamp(min=0)),
            'Dry_Total_g':  torch.log1p(total.clamp(min=0)),
            # Original space (for LTN predicates P3-P11 and post-hoc analysis)
            '_green':  green,
            '_dead':   dead,
            '_clover': clover,
            '_gdm':    gdm,
            '_total':  total,
        }
