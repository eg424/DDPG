def update_target(target, source, tau):
    """
    Soft-update target network parameters.
    """
    target_weights = target.get_weights()
    source_weights = source.get_weights()
    updated_weights = [tau * sw + (1 - tau) * tw for sw, tw in zip(source_weights, target_weights)]
    target.set_weights(updated_weights)
