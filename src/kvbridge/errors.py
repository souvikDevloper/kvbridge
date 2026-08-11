"""Domain-specific failures exposed by KVBridge."""


class KVBridgeError(RuntimeError):
    """Base class for all expected KVBridge failures."""


class CompatibilityError(KVBridgeError):
    """Raised when source and target models cannot be transferred safely."""


class CacheValidationError(KVBridgeError):
    """Raised when a cache has an invalid shape or incomplete metadata."""


class ArtifactError(KVBridgeError):
    """Raised when an artifact is malformed, corrupt, or incompatible."""
