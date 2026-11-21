from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class NodeType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    NODE_TYPE_UNSPECIFIED: _ClassVar[NodeType]
    NODE_TYPE_WORKER: _ClassVar[NodeType]
    NODE_TYPE_AGGREGATOR: _ClassVar[NodeType]

class NodeState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    NODE_STATE_UNSPECIFIED: _ClassVar[NodeState]
    NODE_STATE_JOINING: _ClassVar[NodeState]
    NODE_STATE_IDLE: _ClassVar[NodeState]
    NODE_STATE_TRAINING: _ClassVar[NodeState]
    NODE_STATE_UPLOADING: _ClassVar[NodeState]
    NODE_STATE_AGGREGATING: _ClassVar[NodeState]
NODE_TYPE_UNSPECIFIED: NodeType
NODE_TYPE_WORKER: NodeType
NODE_TYPE_AGGREGATOR: NodeType
NODE_STATE_UNSPECIFIED: NodeState
NODE_STATE_JOINING: NodeState
NODE_STATE_IDLE: NodeState
NODE_STATE_TRAINING: NodeState
NODE_STATE_UPLOADING: NodeState
NODE_STATE_AGGREGATING: NodeState

class NodeMsg(_message.Message):
    __slots__ = ("register", "describe", "heartbeat", "state")
    REGISTER_FIELD_NUMBER: _ClassVar[int]
    DESCRIBE_FIELD_NUMBER: _ClassVar[int]
    HEARTBEAT_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    register: Register
    describe: DescribeResponse
    heartbeat: Heartbeat
    state: StateUpdate
    def __init__(self, register: _Optional[_Union[Register, _Mapping]] = ..., describe: _Optional[_Union[DescribeResponse, _Mapping]] = ..., heartbeat: _Optional[_Union[Heartbeat, _Mapping]] = ..., state: _Optional[_Union[StateUpdate, _Mapping]] = ...) -> None: ...

class CoordinatorMsg(_message.Message):
    __slots__ = ("register_ack", "describe", "new_checkpoint", "aggregator", "shutdown")
    REGISTER_ACK_FIELD_NUMBER: _ClassVar[int]
    DESCRIBE_FIELD_NUMBER: _ClassVar[int]
    NEW_CHECKPOINT_FIELD_NUMBER: _ClassVar[int]
    AGGREGATOR_FIELD_NUMBER: _ClassVar[int]
    SHUTDOWN_FIELD_NUMBER: _ClassVar[int]
    register_ack: RegisterAck
    describe: DescribeRequest
    new_checkpoint: NewCheckpoint
    aggregator: AggregatorInfo
    shutdown: Shutdown
    def __init__(self, register_ack: _Optional[_Union[RegisterAck, _Mapping]] = ..., describe: _Optional[_Union[DescribeRequest, _Mapping]] = ..., new_checkpoint: _Optional[_Union[NewCheckpoint, _Mapping]] = ..., aggregator: _Optional[_Union[AggregatorInfo, _Mapping]] = ..., shutdown: _Optional[_Union[Shutdown, _Mapping]] = ...) -> None: ...

class Register(_message.Message):
    __slots__ = ("type", "node_id", "advertise_addr", "version")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    ADVERTISE_ADDR_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    type: NodeType
    node_id: str
    advertise_addr: str
    version: str
    def __init__(self, type: _Optional[_Union[NodeType, str]] = ..., node_id: _Optional[str] = ..., advertise_addr: _Optional[str] = ..., version: _Optional[str] = ...) -> None: ...

class RegisterAck(_message.Message):
    __slots__ = ("node_id", "epoch", "latest_checkpoint_id", "latest_global_version", "node_state_id", "latest_checkpoint_uri")
    NODE_ID_FIELD_NUMBER: _ClassVar[int]
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    LATEST_CHECKPOINT_ID_FIELD_NUMBER: _ClassVar[int]
    LATEST_GLOBAL_VERSION_FIELD_NUMBER: _ClassVar[int]
    NODE_STATE_ID_FIELD_NUMBER: _ClassVar[int]
    LATEST_CHECKPOINT_URI_FIELD_NUMBER: _ClassVar[int]
    node_id: str
    epoch: int
    latest_checkpoint_id: str
    latest_global_version: int
    node_state_id: str
    latest_checkpoint_uri: str
    def __init__(self, node_id: _Optional[str] = ..., epoch: _Optional[int] = ..., latest_checkpoint_id: _Optional[str] = ..., latest_global_version: _Optional[int] = ..., node_state_id: _Optional[str] = ..., latest_checkpoint_uri: _Optional[str] = ...) -> None: ...

class DescribeRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class DescribeResponse(_message.Message):
    __slots__ = ("h", "inner_lr", "batch_size", "device", "extra")
    class ExtraEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    H_FIELD_NUMBER: _ClassVar[int]
    INNER_LR_FIELD_NUMBER: _ClassVar[int]
    BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    EXTRA_FIELD_NUMBER: _ClassVar[int]
    h: int
    inner_lr: float
    batch_size: int
    device: str
    extra: _containers.ScalarMap[str, str]
    def __init__(self, h: _Optional[int] = ..., inner_lr: _Optional[float] = ..., batch_size: _Optional[int] = ..., device: _Optional[str] = ..., extra: _Optional[_Mapping[str, str]] = ...) -> None: ...

class Heartbeat(_message.Message):
    __slots__ = ("state", "base_global_version", "inner_step", "loss", "samples_per_sec", "gpu_util")
    STATE_FIELD_NUMBER: _ClassVar[int]
    BASE_GLOBAL_VERSION_FIELD_NUMBER: _ClassVar[int]
    INNER_STEP_FIELD_NUMBER: _ClassVar[int]
    LOSS_FIELD_NUMBER: _ClassVar[int]
    SAMPLES_PER_SEC_FIELD_NUMBER: _ClassVar[int]
    GPU_UTIL_FIELD_NUMBER: _ClassVar[int]
    state: NodeState
    base_global_version: int
    inner_step: int
    loss: float
    samples_per_sec: float
    gpu_util: float
    def __init__(self, state: _Optional[_Union[NodeState, str]] = ..., base_global_version: _Optional[int] = ..., inner_step: _Optional[int] = ..., loss: _Optional[float] = ..., samples_per_sec: _Optional[float] = ..., gpu_util: _Optional[float] = ...) -> None: ...

class StateUpdate(_message.Message):
    __slots__ = ("node_state_id",)
    NODE_STATE_ID_FIELD_NUMBER: _ClassVar[int]
    node_state_id: str
    def __init__(self, node_state_id: _Optional[str] = ...) -> None: ...

class NewCheckpoint(_message.Message):
    __slots__ = ("checkpoint_id", "global_version", "uri")
    CHECKPOINT_ID_FIELD_NUMBER: _ClassVar[int]
    GLOBAL_VERSION_FIELD_NUMBER: _ClassVar[int]
    URI_FIELD_NUMBER: _ClassVar[int]
    checkpoint_id: str
    global_version: int
    uri: str
    def __init__(self, checkpoint_id: _Optional[str] = ..., global_version: _Optional[int] = ..., uri: _Optional[str] = ...) -> None: ...

class AggregatorInfo(_message.Message):
    __slots__ = ("addr", "epoch")
    ADDR_FIELD_NUMBER: _ClassVar[int]
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    addr: str
    epoch: int
    def __init__(self, addr: _Optional[str] = ..., epoch: _Optional[int] = ...) -> None: ...

class Shutdown(_message.Message):
    __slots__ = ("reason",)
    REASON_FIELD_NUMBER: _ClassVar[int]
    reason: str
    def __init__(self, reason: _Optional[str] = ...) -> None: ...

class DeltaRecord(_message.Message):
    __slots__ = ("delta_id", "worker_id", "base_global_version", "inner_steps", "num_samples", "status", "reason")
    DELTA_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    BASE_GLOBAL_VERSION_FIELD_NUMBER: _ClassVar[int]
    INNER_STEPS_FIELD_NUMBER: _ClassVar[int]
    NUM_SAMPLES_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    delta_id: str
    worker_id: str
    base_global_version: int
    inner_steps: int
    num_samples: int
    status: str
    reason: str
    def __init__(self, delta_id: _Optional[str] = ..., worker_id: _Optional[str] = ..., base_global_version: _Optional[int] = ..., inner_steps: _Optional[int] = ..., num_samples: _Optional[int] = ..., status: _Optional[str] = ..., reason: _Optional[str] = ...) -> None: ...

class CommitCheckpointRequest(_message.Message):
    __slots__ = ("aggregator_id", "epoch", "checkpoint_id", "uri", "parent_version", "delta_ids", "delta_records", "initial")
    AGGREGATOR_ID_FIELD_NUMBER: _ClassVar[int]
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINT_ID_FIELD_NUMBER: _ClassVar[int]
    URI_FIELD_NUMBER: _ClassVar[int]
    PARENT_VERSION_FIELD_NUMBER: _ClassVar[int]
    DELTA_IDS_FIELD_NUMBER: _ClassVar[int]
    DELTA_RECORDS_FIELD_NUMBER: _ClassVar[int]
    INITIAL_FIELD_NUMBER: _ClassVar[int]
    aggregator_id: str
    epoch: int
    checkpoint_id: str
    uri: str
    parent_version: int
    delta_ids: _containers.RepeatedScalarFieldContainer[str]
    delta_records: _containers.RepeatedCompositeFieldContainer[DeltaRecord]
    initial: bool
    def __init__(self, aggregator_id: _Optional[str] = ..., epoch: _Optional[int] = ..., checkpoint_id: _Optional[str] = ..., uri: _Optional[str] = ..., parent_version: _Optional[int] = ..., delta_ids: _Optional[_Iterable[str]] = ..., delta_records: _Optional[_Iterable[_Union[DeltaRecord, _Mapping]]] = ..., initial: _Optional[bool] = ...) -> None: ...

class CommitCheckpointResponse(_message.Message):
    __slots__ = ("accepted", "global_version", "reason")
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    GLOBAL_VERSION_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    accepted: bool
    global_version: int
    reason: str
    def __init__(self, accepted: _Optional[bool] = ..., global_version: _Optional[int] = ..., reason: _Optional[str] = ...) -> None: ...

class DeltaChunk(_message.Message):
    __slots__ = ("header", "seq", "data", "last")
    HEADER_FIELD_NUMBER: _ClassVar[int]
    SEQ_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    LAST_FIELD_NUMBER: _ClassVar[int]
    header: DeltaHeader
    seq: int
    data: bytes
    last: bool
    def __init__(self, header: _Optional[_Union[DeltaHeader, _Mapping]] = ..., seq: _Optional[int] = ..., data: _Optional[bytes] = ..., last: _Optional[bool] = ...) -> None: ...

class DeltaHeader(_message.Message):
    __slots__ = ("delta_id", "worker_id", "base_checkpoint_id", "base_global_version", "inner_steps", "num_samples", "local_loss", "total_bytes", "sha256")
    DELTA_ID_FIELD_NUMBER: _ClassVar[int]
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    BASE_CHECKPOINT_ID_FIELD_NUMBER: _ClassVar[int]
    BASE_GLOBAL_VERSION_FIELD_NUMBER: _ClassVar[int]
    INNER_STEPS_FIELD_NUMBER: _ClassVar[int]
    NUM_SAMPLES_FIELD_NUMBER: _ClassVar[int]
    LOCAL_LOSS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_BYTES_FIELD_NUMBER: _ClassVar[int]
    SHA256_FIELD_NUMBER: _ClassVar[int]
    delta_id: str
    worker_id: str
    base_checkpoint_id: str
    base_global_version: int
    inner_steps: int
    num_samples: int
    local_loss: float
    total_bytes: int
    sha256: str
    def __init__(self, delta_id: _Optional[str] = ..., worker_id: _Optional[str] = ..., base_checkpoint_id: _Optional[str] = ..., base_global_version: _Optional[int] = ..., inner_steps: _Optional[int] = ..., num_samples: _Optional[int] = ..., local_loss: _Optional[float] = ..., total_bytes: _Optional[int] = ..., sha256: _Optional[str] = ...) -> None: ...

class PushDeltaAck(_message.Message):
    __slots__ = ("accepted", "reason", "latest_checkpoint_id", "latest_global_version", "latest_checkpoint_uri")
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    LATEST_CHECKPOINT_ID_FIELD_NUMBER: _ClassVar[int]
    LATEST_GLOBAL_VERSION_FIELD_NUMBER: _ClassVar[int]
    LATEST_CHECKPOINT_URI_FIELD_NUMBER: _ClassVar[int]
    accepted: bool
    reason: str
    latest_checkpoint_id: str
    latest_global_version: int
    latest_checkpoint_uri: str
    def __init__(self, accepted: _Optional[bool] = ..., reason: _Optional[str] = ..., latest_checkpoint_id: _Optional[str] = ..., latest_global_version: _Optional[int] = ..., latest_checkpoint_uri: _Optional[str] = ...) -> None: ...
