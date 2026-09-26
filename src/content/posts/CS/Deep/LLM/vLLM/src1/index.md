---
title: "vLLM 源码: 初始化 [LLM]"
published: 2026-09-24 09:00:00
category: CS
image: ./cover.png
---

`vLLM` 的初始化主要涉及以下几ß个组件：
- `uvicorn`：负责接收请求的服务器
- `AsyncLLM`：前端网关
- `EngineCoreClient`：负责 `AsyncLLM` 和 `EngineCore` 间的通信
- `EngineCore`: 请求调度核心，`KVCache` 管理等
- `Executor`: 加载权重，执行推理等
- `Scheduler`

## 1 准备工作

通过 `vllm serve` 命令启动服务：

```python
vllm serve Qwenxxx --host 0.0.0.0 --port 8000
```

主类在 `vllm/vllm/cli/main.py`:

```python
# main.py
def main():
	# 懒加载
	import vllm.entrypoints.cli.openai
    import vllm.entrypoints.cli.run_batch
    import vllm.entrypoints.cli.serve
	...
	from vllm.utils.argparse_utils import FlexibleArgumentParser

	# 存放到 CMD_MODULES 中
	CMD_MODULES = [
        vllm.entrypoints.cli.openai,
        vllm.entrypoints.cli.serve,
        ...
    ]

	...
	subparsers = parser.add_subparsers(required=False, dest="subparser")
	cmds = {}
    for cmd_module in CMD_MODULES:
        if cmd_module is vllm.entrypoints.cli.snapshot: ...
		# 遍历 CMD_MODULES 列表调用每个模块的 .cmd_init()，将子命令注册到 argparse 子解析器
        else:
			# 不同的模块返回的实例个数不同
            new_cmds = cmd_module.cmd_init()
        for cmd in new_cmds:
			# 调用实例的 subparser_init() 初始化 subparser
            cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
            cmds[cmd.name] = cmd
    args = parser.parse_args()
    if args.subparser in cmds:
        cmds[args.subparser].validate(args)
```

`serve.py` 也在 `cil` 文件夹下:

```python
# serve.py
# serve 的 init() 会返回含有单个 CLISubcommand 对象的 list
def cmd_init() -> list[CLISubcommand]:
    return [ServeSubcommand()]

class ServeSubcommand(CLISubcommand):
    """The `serve` subcommand for the vLLM CLI."""
    name = "serve"

	@staticmethod
    def cmd(args: argparse.Namespace) -> None:
		# 指定模型
		if hasattr(args, "model_tag") and args.model_tag is not None:
            args.model = args.model_tag

		# 是否使用 vLLM 的高性能前端而非 uvicorn
		rust_frontend_path = (envs.VLLM_RUST_FRONTEND_PATH if envs.VLLM_USE_RUST_FRONTEND else None)

		...

		# 计算启动 API 实例的个数，未指定时进行计算
		if args.api_server_count is None: ...

		if is_multi_port: ...
        elif args.api_server_count < 1: ...
        elif args.api_server_count > 1 or rust_frontend_path: ...
        else:
            # Single API server (this process).
            args.api_server_count = None
			# 启动服务器
			# uvloop 用来跑异步 IO 代码
            uvloop.run(run_server(args))
```

`run_server()` 在 `vllm/entrypoints/launchers/api_server/entry.py` 下：

```python
# entry.py
async def run_server(args, **uvicorn_kwargs) -> None:
    """Run a single-worker API server."""
	# 收到 SIGTERM 信号的话执行 _interrupt_init
    signal.signal(signal.SIGTERM, _interrupt_init)
    listen_address, sock = setup_server(args, reuse_port=False)
    await run_server_worker(listen_address, sock, args, **uvicorn_kwargs)
```

- `setup_server()` 负责参数校验，创建绑定 `socket`...

```python
# entry.py
def setup_server(args, *, reuse_port: bool):
	if args.uds:
        sock = create_server_unix_socket(args.uds)
    else:
		# 参数中指定的 host 和 port
        sock_addr = (args.host or "", args.port)
        sock = create_server_socket(sock_addr, reuse_port=reuse_port)

	...
	# 返回监听地址和 socket 对象
	return listen_address, sock
```

- `run_server_worker()` 加载工具，创建 `engine_client`:

```python
# entry.py
async def run_server_worker(
	listen_address, sock, args, client_config=None, **uvicorn_kwargs
) -> None:
    """Run a single API server worker."""
	# 加载自定义解析器插件
    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

	# 推理插件
	if args.reasoning_parser_plugin and len(args.reasoning_parser_plugin) > 3:
        ReasoningParserManager.import_reasoning_parser(args.reasoning_parser_plugin)
	# 把 cli arg 转成 engine args，加载模型权重
	async with build_async_engine_client(
        args,
        client_config=client_config,
    ) as engine_client:
		# build_async_engine_client() 完成后执行 build_and_serve()
        shutdown_task = await build_and_serve(
            engine_client, listen_address, sock, args, **uvicorn_kwargs
        )
```

## 2 启动组件
### 2.1 实例化 AsyncLLM
`AsyncLLM` 是 `Python API` 前端的异步请求网关

```python
# entry.py
async def build_async_engine_client(
    args: Namespace,
    *,
    usage_context: UsageContext = UsageContext.OPENAI_API_SERVER,
    client_config: dict[str, Any] | None = None,
) -> AsyncIterator[EngineClient]:

    # Context manager to handle engine_client lifecycle
    # Ensures everything is shutdown and cleaned up on error/exit
	# 从命令行中解析出来的 args 提取需要的字段生成 AsyncEngineArgs 实例
    engine_args = AsyncEngineArgs.from_cli_args(args)
	# 只有在多 API 模式下才不为 None
    if client_config: ...

    async with build_async_engine_client_from_engine_args(
        engine_args,
        usage_context=usage_context,
        client_config=client_config,
    ) as engine:
        yield engine
```

`build_async_engine_client_from_engine_args()` 创建 `VllmConfig` 和 `AsyncLLM`:

```python
async def build_async_engine_client_from_engine_args(
    engine_args: AsyncEngineArgs,
    *,
    usage_context: UsageContext = UsageContext.OPENAI_API_SERVER,
    client_config: dict[str, Any] | None = None,
) -> AsyncIterator[EngineClient]:
    """Create EngineClient, either:
        - in-process using the AsyncLLMEngine Directly
        - multiprocess using AsyncLLMEngine RPC

    Returns the Client or None if the creation failed.
    """
	# 创建 VllmConfig 对象
	# vLLM 引擎的总配置容器，由 ModelConfig, CacheConfig, ParallelConfig... 等多个子配置对象组成
    # Create the EngineConfig (determines if we can use V1).
    vllm_config = engine_args.create_engine_config(usage_context=usage_context)

	# v1 引擎对外的主入口类
    from vllm.v1.engine.async_llm import AsyncLLM

    async_llm: AsyncLLM | None = None

    try:
		# 通过静态方法实例化 AsyncLLM
        async_llm = AsyncLLM.from_vllm_config(
            vllm_config=vllm_config,
            usage_context=usage_context,
            enable_log_requests=engine_args.enable_log_requests,
            aggregate_engine_logging=engine_args.aggregate_engine_logging,
            disable_log_stats=engine_args.disable_log_stats,
            client_addresses=client_config,
            client_count=client_count,
            client_index=client_index,
        )

        # Don't keep the dummy data in memory
        assert async_llm is not None
        await async_llm.reset_mm_cache()

        yield async_llm
    finally:
        if async_llm:
            async_llm.shutdown(timeout=vllm_config.shutdown_timeout)


@classmethod
def from_vllm_config(
	cls,
	vllm_config: VllmConfig,
	start_engine_loop: bool = True,
	usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
	stat_loggers: list[StatLoggerFactory] | None = None,
	enable_log_requests: bool = False,
	aggregate_engine_logging: bool = False,
	disable_log_stats: bool = False,
	client_addresses: dict[str, Any] | None = None,
	client_count: int = 1,
	client_index: int = 0,
) -> "AsyncLLM":
	# Create the LLMEngine.
	# cls 表示当前类的实例，也就是调构造方法
	# 获取 Executor 的类名
	return cls(
		vllm_config=vllm_config,
		executor_class=Executor.get_class(vllm_config),
		start_engine_loop=start_engine_loop,
		stat_loggers=stat_loggers,
		log_requests=enable_log_requests,
		log_stats=not disable_log_stats,
		aggregate_engine_logging=aggregate_engine_logging,
		usage_context=usage_context,
		client_addresses=client_addresses,
		client_count=client_count,
		client_index=client_index,
	)
```

`AsyncLLM` 类在 `vllm/v1/engine/async_llm.py` 下，创建 `InputProcessor` 和 `OutputProcessor` 用于处理请求，创建 `AsyncMPClient` 进行通信

```python
# async_llm.py
class AsyncLLM(EngineClient):

	def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        log_requests: bool = True,
        start_engine_loop: bool = True,
        stat_loggers: list[StatLoggerFactory] | None = None,
        aggregate_engine_logging: bool = False,
        client_addresses: dict[str, Any] | None = None,
        client_count: int = 1,
        client_index: int = 0,
        profiler: TorchProfilerWrapper | None = None,
    ) -> None:
        # Create an AsyncLLM.

        # Ensure we can serialize custom transformer configs
        maybe_register_config_serialize_by_value()

        self.vllm_config = vllm_config
        ...
        self.model_config = vllm_config.model_config
        self.scheduler_config = vllm_config.scheduler_config

		# 处理 prompt
        self.renderer = renderer = renderer_from_config(self.vllm_config)

		# 将上层 api 收到的 EngineInput 转换成引擎内核可以识别的 EngineCoreRequest
        # Convert EngineInput --> EngineCoreRequest.
        self.input_processor = InputProcessor(self.vllm_config, renderer)

		# 把引擎输出转换成用户可以看懂的
        # Converts EngineCoreOutputs --> RequestOutput.
        self.output_processor = OutputProcessor(
            renderer.tokenizer,
            log_stats=self.log_stats,
            stream_interval=self.vllm_config.scheduler_config.stream_interval,
            tracing_enabled=tracing_endpoint is not None,
            admission_stats=self.admission_stats,
        )

        # EngineCore (starts the engine in background process).
        # Hand the renderer to the client so it can start the frontend MM
        # warmup only after engine-core fork (the why is in
        # BaseRenderer.start_mm_warmup_in_background). The warmup is joined
        # by reset_mm_cache / warmup / shutdown.
		# 创建 EngineCoreClient
        self.engine_core = EngineCoreClient.make_async_mp_client(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=self.log_stats,
            client_addresses=client_addresses,
            client_count=client_count,
            client_index=client_index,
            renderer=renderer,
        )
        ...
```

### 2.2 AsyncMPClient 及父类的实例化
`EngineCoreClient` 是 `AsyncLLM ↔ EngineCore` 的 `IPC` 通信客户端，`AsyncMPClient` 是其中一个实现，对应文件在 `vllm/v1/engine/core_client.py` 下:

```python
# core_client.py
class EngineCoreClient(ABC):
	def make_async_mp_client(
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_addresses: dict[str, Any] | None = None,
        client_count: int = 1,
        client_index: int = 0,
        renderer: BaseRenderer | None = None,
    ) -> "AsyncMPClient":
        parallel_config = vllm_config.parallel_config
        client_args = (
            vllm_config,
            executor_class,
            log_stats,
            client_addresses,
            client_count,
            client_index,
        )
        if parallel_config.data_parallel_size > 1: ...
		# 返回一个 AsyncMPClient
        return AsyncMPClient(
            *client_args,
            renderer=renderer,
        )
```

```python
# core_client.py
# 继承了 MPClient
class AsyncMPClient(MPClient):
    """Asyncio-compatible client for multi-proc EngineCore."""

    @instrument(span_name="AsyncMPClient init")
    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_addresses: dict[str, Any] | None = None,
        client_count: int = 1,
        client_index: int = 0,
        renderer: BaseRenderer | None = None,
    ):
        super().__init__(
            asyncio_mode=True,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=log_stats,
            client_addresses=client_addresses,
            renderer=renderer,
        )

		# 消息队列，交给 OutputProcessor 解析
		self.outputs_queue = asyncio.Queue[EngineCoreOutputs | Exception]()

		...
		try:
            # If we are running in an asyncio event loop, start the queue task.
            # Otherwise, it will be started lazily. If it is not started here,
            # we could miss EXECUTOR_FAILED messages from engine core if they
            # occur prior to any requests being sent.
            asyncio.get_running_loop()
			# 启动后台队列消费任务，循环从 ZMQ socket 读取 EngineCore 返回的消息，放入self.outputs_queue
            self._ensure_output_queue_task()
        except RuntimeError:
            pass
```

模型加载、`GPU worker` 启动在父类 `MPClient` 的初始化方法中:

```python
# core_client.py
class MPClient(EngineCoreClient):
	def __init__(
        self,
        asyncio_mode: bool,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_addresses: dict[str, Any] | None = None,
        renderer: BaseRenderer | None = None,
    ):
		self.vllm_config = vllm_config
        self._renderer: BaseRenderer | None = renderer
        self._effective_attention_block_sizes: set[int | None] = set()
        self._kv_event_sources: dict[int, KVEventsConfig] = {}

        # ZMQ setup.
		# 2 个后台 IO 线程处理消息收发
        sync_ctx = zmq.Context(io_threads=2)
        self.ctx = zmq.asyncio.Context(sync_ctx) if asyncio_mode else sync_ctx

		try:
            # State used for data parallel.
            self.engines_running = False
			...
			if client_addresses: # 单机是 None
			else:
                # Engines are managed by this client.
                addresses = get_engine_zmq_addresses(vllm_config)
                self.input_socket = self.resources.input_socket = make_zmq_socket(
                    self.ctx,
                    addresses.inputs[0],
                    zmq.ROUTER,
                    bind=True,
                    router_handover=enable_input_socket_handover,
                )
                self.resources.output_socket = make_zmq_socket(
                    self.ctx, addresses.outputs[0], zmq.PULL
                )

				# 将占位地址解析成真实监听地址
                # Resolve tcp://host:0 placeholders to bound endpoints
                # before engines DEALER-connect. No-op for IPC.
                addresses.inputs[0] = self.input_socket.getsockopt(
                    zmq.LAST_ENDPOINT
                ).decode()
                addresses.outputs[0] = self.resources.output_socket.getsockopt(
                    zmq.LAST_ENDPOINT
                ).decode()
				# 下一步的逻辑
				with launch_core_engines(
                    vllm_config, executor_class, log_stats, addresses
                ) as engine_launch:
                    self.resources.coordinator = engine_launch.coordinator
                    self.resources.engine_manager = engine_launch.engine_manager
                    coordinator = engine_launch.coordinator
                    addresses = engine_launch.addresses
                    tensor_queue = engine_launch.tensor_queue
                    # Engine-core processes have now all been forked/started
                    # (CoreEngineProcManager.proc.start()). It is now safe to
                    # launch the frontend background MM warmup: it must not
                    # run while fork() is in flight (a live thread holding a
                    # lock would deadlock the forked child), but the
                    # engine-core model load (minutes) that follows is exactly
                    # what the warmup should overlap with.
                    self._start_mm_warmup()
			
```

### 2.3 实例化 CoreEngineProcManager，EngineCore 进程
`launch_core_engines()` 创建 `CoreEngineProcManager`:

```python
# utils.py
def launch_core_engines(
	vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool,
    addresses: EngineZmqAddresses,
) -> Iterator[CoreEngineLaunch]:
	...
	with zmq_socket_ctx(
        local_handshake_address, zmq.ROUTER, bind=True
    ) as handshake_socket:
        # Start local engines.
        if local_engine_count:
            local_engine_manager = CoreEngineProcManager(
                vllm_config=vllm_config,
                executor_class=executor_class,
                log_stats=log_stats,
                handshake_address=handshake_address,
                client_handshake_address=client_handshake_address,
                local_client=True,
                local_engine_count=local_engine_count,
                start_index=dp_rank,
                local_start_index=local_start_index or 0,
                tensor_queue=tensor_queue,
            )
        else:
            local_engine_manager = None

		launch = CoreEngineLaunch(
            local_engine_manager, coordinator, addresses, tensor_queue
        )
		# 暂停回到 MPClient，MPClient执行 self._start_mm_warmup()
		# 生成的 CoreEngineLaunch 返回给 AsyncMPClient
        yield launch
		# 回到这里执行 wait_for_engine_startup，阻塞等待 EngineCore 的 READY 握手消息
		wait_for_engine_startup(
            handshake_socket,
            engines_to_handshake,
            parallel_config,
            dp_size > 1 and vllm_config.model_config.is_moe,
            vllm_config.cache_config,
            launch,
        )
```

`CoreEngineProcManager` 的构造方法在 `vllm/v1/engine/utils.py` 下：

```python
class CoreEngineProcManager:
	def __init__(
        self,
		# 启动多少个 EngineCore 进程
        local_engine_count: int,
        start_index: int,
        local_start_index: int,
        vllm_config: VllmConfig,
        local_client: bool,
		# ZMQ 握手地址，用于 client 和 engine core 通信
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: str | None = None,
        tensor_queue: Queue | None = None,
    ):
        self._request_shutdown_timeout = vllm_config.shutdown_timeout
		# 从环境变量 VLLM_WORKER_MULTIPROC_METHOD 中获取启动方法 (spawn/fork/forkserver)
        context = get_mp_context()
		# 公共参数字典
        common_kwargs = {
            "vllm_config": vllm_config,
            "local_client": local_client,
            "handshake_address": handshake_address,
            "executor_class": executor_class,
            "log_stats": log_stats,
            "tensor_queue": tensor_queue,
        }

        if client_handshake_address:
            common_kwargs["client_handshake_address"] = client_handshake_address

		# 是否并行
        is_dp = vllm_config.parallel_config.data_parallel_size > 1

        from vllm.v1.engine.core import EngineCoreProc

        self.processes: list[BaseProcess] = []
        local_dp_ranks  = []
		# 循环创建 EngineCore 进程对象
		# vLLM 推理系统的调度与 KV 缓存管理核心
        for index in range(local_engine_count):
            local_index = local_start_index + index
            global_index = start_index + index

            # Start EngineCore in background process.
            local_dp_ranks.append(local_index)
            self.processes.append(
				# 根据上下文类型创建对应的 Process 对象，默认 spawn
                context.Process(
                    target=EngineCoreProc.run_engine_core,
                    name=f"EngineCore_DP{global_index}" if is_dp else "EngineCore",
                    kwargs=common_kwargs
                    | {"dp_rank": global_index, "local_dp_rank": local_index},
                )
            )
		# 当这个管理器实例被 GC 回收时，自动调用 shutdown，杀掉所有 EngineCore 子进程，防止进程泄漏
        self._finalizer = weakref.finalize(self, shutdown, self.processes)
        self.manager_stopped = threading.Event()
        self.failed_proc_name: str | None = None

        # All ranks share this config object: capture the user-provided
        # --device-ids list before the per-rank shard overwrites it. Mutating
        # the config before each proc.start() works because the spawn method
        # pickles process args at start() time, sequentially per rank.
		# numa(Non-Uniform Memory Access) 指在多 CPU服务器上：每个 CPU 插槽，自带一块本地内存
        user_assigned_gpu_ids = vllm_config.parallel_config.assigned_physical_gpu_ids
        try:
			# 循环启动所有子进程 + GPU/NUMA 绑定
            for proc, local_dp_rank in zip(self.processes, local_dp_ranks):
                # Populate the logical-to-physical GPU mapping in DP for
                # platforms that cannot rely on
                # torch.accelerator.set_device_index(), and for Ray.
                needs_device_env_isolation = not (
                    current_platform.is_cuda_alike() or current_platform.is_xpu()
                )
                if is_dp and (
                    needs_device_env_isolation or vllm_config.parallel_config.use_ray
                ):
                    set_assigned_physical_gpu_ids_for_dp_rank(
                        vllm_config, local_dp_rank, user_assigned_gpu_ids
                    )

                with numa_utils.configure_subprocess(
                    # EngineCore itself does not have a TP/PP-local rank.
                    # When DP is enabled, set_assigned_physical_gpu_ids_for_dp_rank()
                    # populates the logical-to-physical mapping for this DP
                    # shard, so local_rank=0 means "the first local GPU in
                    # this shard". The actual TP/PP worker processes spawned
                    # by the executor are bound separately with their own
                    # local_rank values.
                    vllm_config,
                    local_rank=0,
                    dp_local_rank=local_dp_rank,
                    process_kind="EngineCore",
                ):
					# 启动
                    proc.start()
        finally:
            # Kill other procs if not all are running.
            if self.finished_procs():
                self.shutdown()
```

### 2.4 启动 EngineCore 进程

子进程启动后，`multiprocessing` 框架内部调用 `run()`，执行 `context.Process` 中的 `target: run_engine_core()`

```python
# core.py
class EngineCoreProc(EngineCore):
	 """ZMQ-wrapper for running **EngineCore** in background process."""
	 ...
	@staticmethod
    def run_engine_core(*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs):
        """Launch EngineCore busy loop in background process."""
        # Ensure we can serialize transformer config after spawning
        maybe_register_config_serialize_by_value()

        engine_core: EngineCoreProc | None = None
        signal_callback: SignalCallback | None = None
        clean_shutdown = False
        try:
            vllm_config: VllmConfig = kwargs["vllm_config"]
            parallel_config: ParallelConfig = vllm_config.parallel_config
            data_parallel = parallel_config.data_parallel_size > 1 or dp_rank > 0
			# 取出全局配置，判断是否开启 DP
            if data_parallel:
                parallel_config.data_parallel_rank_local = local_dp_rank
                process_title = f"EngineCore_DP{dp_rank}"
            else:
                process_title = "EngineCore"
            set_process_title(process_title)
            maybe_init_worker_tracer("vllm.engine_core", "engine_core", process_title)
            
            parallel_config.data_parallel_index = dp_rank
			# 不开启 MoE 实例化 EngineCoreProc
            if data_parallel and vllm_config.model_config.is_moe:
                # Set data parallel rank for this engine process.
                parallel_config.data_parallel_rank = dp_rank
                engine_core = DPEngineCoreProc(*args, **kwargs)
            else:
                # Non-MoE DP ranks are completely independent, so treat like DP=1.
                # Note that parallel_config.data_parallel_index will still reflect
                # the original DP rank.
                parallel_config.reconfigure_for_independent_dp_rank()
                engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)

            assert engine_core is not None

			# 调用 EngineCoreProc 的 run_busy_loop()
            engine_core.run_busy_loop()

		# 无论正常退出还是崩溃，一定会调用 engine_core.shutdown()
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            if signal_callback is not None:
                signal_callback.stop()
            if engine_core is not None:
                engine_core.shutdown()
            if clean_shutdown:
                from vllm.platforms import current_platform

                if current_platform.is_rocm():
                    # Cleanup above already unfreezes and collects the heap.
                    # Freeze the surviving graph to skip another slow cyclic-GC
                    # scan during finalization; process exit reclaims it.
                    gc.freeze()

	def run_busy_loop(self):
        """Core busy loop of the EngineCore."""
        while self._handle_shutdown():
            # 1) Poll the input queue until there is work to do.
			# 检查关闭状态
            self._process_input_queue()
            # Publish request counts before and after GPU step to ensure freshness.
			# 从 ZMQ / 输入队列读取 API 侧发来的请求
            self._maybe_publish_request_counts()
            # 2) Step the engine core and return the outputs.
			# 一轮完整的引擎 step，核心计算调度
            self._process_engine_step()
            self._maybe_publish_request_counts()

        raise SystemExit

	# 调度器选出一批可以跑的 request, 做批处理调度
	# 把这个 batch 任务下发给 Executor
	# Executor 转发任务到 GPU Worker，执行 prefill 或者 decode 前向计算
	# GPU Worker 计算完成，返回生成 token、更新后的 KV 信息
	# EngineCore 收集结果，组装输出，通过 ZMQ 发回 API 客户端
	def _process_engine_step(self) -> bool:
        """Called only when there are unfinished local requests."""
        # Step the engine core.
        outputs, model_executed = self.step_fn()
        # Put EngineCoreOutputs into the output queue.
        for output in outputs.items() if outputs else ():
            self.output_queue.put_nowait(output)
        # Post-step hook.
        self.post_step(model_executed)

        # If no model execution happened but there is still scheduler work
        # (e.g. WAITING_FOR_REMOTE_KVS or delayed KV connector frees), yield
        # the GIL briefly to allow background transfer threads to make progress.
        if not model_executed and self.scheduler.has_requests():
            time.sleep(0.001)

        return model_executed
```

#### 2.4.1 EngineCoreProc 的实例化：
`EngineCoreProc` 继承自 `EngineCore`，`EngineCore` 是内核业务逻辑父类

```python
# core.py
class EngineCoreProc(EngineCore):
	@instrument(span_name="EngineCoreProc init")
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: str | None = None,
        tensor_queue: Queue | None = None,
        *,
        engine_index: int = 0,
    ):
		# run_busy_loop 使用的内存队列
        self.input_queue = queue.Queue[tuple[EngineCoreRequestType, Any]]()
        self.output_queue = queue.Queue[tuple[int, EngineCoreOutputs] | bytes]()
		# GPU 崩溃时，往 input 队列塞一个失败事件，主循环感知到后做故障处理
        executor_fail_callback = lambda: self.input_queue.put_nowait(
            (EngineCoreRequestType.EXECUTOR_FAILED, b"")
        )


		# EngineCore 进程与 AsyncMPClient，DP Coordinator 完成 ZMQ socket 握手
		# 拿到解析好的真实监听地址
        with self._perform_handshakes(
            handshake_address,
            identity,
            local_client,
            vllm_config,
            client_handshake_address,
        ) as addresses:
            # Set up data parallel environment.
            self.has_coordinator = addresses.coordinator_output is not None
            self.frontend_stats_publish_address = (
                addresses.frontend_stats_publish_address
            )
            logger.debug(
                "Has DP Coordinator: %s, stats publish address: %s",
                self.has_coordinator,
                self.frontend_stats_publish_address,
            )
            internal_dp_balancing = (
                self.has_coordinator
                and not vllm_config.parallel_config.data_parallel_external_lb
            )
            # Only publish request queue stats to coordinator for "internal"
            # and "hybrid" LB modes.
            self.publish_dp_lb_stats = internal_dp_balancing
            self.last_counts = (0, 0)

            self.addresses = addresses
            self.process_input_queue_block = True
            self._init_data_parallel(vllm_config)

			# 调用 EngineCore 的实例化方法，创建 Executor
            super().__init__(
                vllm_config,
                executor_class,
                log_stats,
                executor_fail_callback,
                internal_dp_balancing,
            )

            # Background Threads and Queues for IO. These enable us to
            # overlap ZMQ socket IO with GPU since they release the GIL,
            # and to overlap some serialization/deserialization with the
            # model forward pass.
            # Threads handle Socket <-> Queues and core_busy_loop uses Queue.
            ready_event = threading.Event()
			# input_thread 读 ZMQ socket，收到请求后，put 进 input_queue
            input_thread = threading.Thread(
                target=self.process_input_sockets,
                args=(
                    addresses.inputs,
                    addresses.coordinator_input,
                    identity,
                    ready_event,
                ),
                daemon=True,
            )
            input_thread.start()

            self.output_thread = threading.Thread(
                target=self.process_output_sockets,
                args=(
                    addresses.outputs,
                    addresses.coordinator_output,
                    self.engine_index,
                ),
                daemon=True,
            )
            self.output_thread.start()

            # Don't complete handshake until DP coordinator ready message is
            # received.
            while not ready_event.wait(timeout=10):
                if not input_thread.is_alive():
                    raise RuntimeError("Input socket thread died during startup")
                assert addresses.coordinator_input is not None
                logger.info("Waiting for READY message from DP Coordinator...")
```

#### 2.4.2 EngineCore 实例化，创建 Executor

`EngineCore` 负责调度，`KV` 缓存，`batch` 推理:

```python
# core.py
class EngineCore:
	"""Inner loop of vLLM's Engine."""

	def __init__(
        self,
        vllm_config: VllmConfig,
		# Executor 的抽象基类
        executor_class: type[Executor],
        log_stats: bool,
        executor_fail_callback: Callable | None = None,
        include_finished_set: bool = False,
    ):
		from vllm.plugins import load_general_plugins
		# 加载插件
        load_general_plugins()

		# VllmConfig 中通过 get_class() 获取对应的 Executor 类
		# executor_class = Executor.get_class(self)
		# 根据配置项 parallel_config.distributed_executor_backend 返回对应的类
		# "mp" == MultiprocExecutor
		self.model_executor = executor_class(vllm_config)
		...

		# 初始化 KVCache
		kv_cache_config = self._initialize_kv_caches(vllm_config)
        self.structured_output_manager = StructuredOutputManager(vllm_config)

		# Scheduler 的实例化
		Scheduler = vllm_config.scheduler_config.get_scheduler_cls()
		...
		self.scheduler: SchedulerInterface = Scheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=self.structured_output_manager,
            include_finished_set=include_finished_set,
            log_stats=self.log_stats,
            block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
        )

		...
		# Batch Queue 流水线并行优化
		self.batch_queue_size = vllm_config.max_concurrent_batches
        self.batch_queue: (
            deque[tuple[Future[ModelRunnerOutput], SchedulerOutput, Future[Any]]] | None
        ) = None
        if self.batch_queue_size > 1:
            logger.debug("Batch queue is enabled with size %d", self.batch_queue_size)
            self.batch_queue = deque(maxlen=self.batch_queue_size)
```

## 3 启动 uvicorn
时序梳理：

```python
async def run_server_worker(
	listen_address, sock, args, client_config=None, **uvicorn_kwargs
) -> None:
	async with build_async_engine_client() as engine_client:
        shutdown_task = await build_and_serve(...)

# 进入 build_async_engine_client:
build_async_engine_client(
	async with build_async_engine_client_from_engine_args(...) as engine:
        yield engine
	) as engine_client:

# 进入 build_async_engine_client_from_engine_args：
build_async_engine_client_from_engine_args(
	try:
        async_llm = AsyncLLM.from_vllm_config(...)
        yield async_llm
	) as engine:
	
# AsyncLLM 创建 AsyncMPClient，初始化父类 MPClient 进入 launch_core_engines：
class MPClient(EngineCoreClient):
	def __init__(...):
		try:
			if client_addresses:
			else:
				with launch_core_engines(...) as engine_launch:

# 进入 launch_core_engines：
launch_core_engines(
	with zmq_socket_ctx(...) as handshake_socket:
		launch = CoreEngineLaunch(...)
        yield launch
	) as engine_launch:

# 当 CoreEngine 子进程启动完毕返回 READY 后向上返回
```

最后执行 `build_and_serve()`:

```python
async def build_and_serve(
    engine_client: EngineClient,
    listen_address: str,
    sock: socket.socket,
    args: Namespace,
    **uvicorn_kwargs,
) -> asyncio.Task:
    """Build FastAPI app, initialize state, and start serving.

    Returns the shutdown task for the caller to await.
    """
    # 构建 FastAPI app
    app = build_app(args, supported_tasks, model_config)
    await init_app_state(engine_client, app.state, args, supported_tasks)

    logger.info("Starting vLLM server on %s", listen_address)

	# 启动 uvicorn http 服务
    return await serve_http(
        app,
        sock=sock,
        enable_ssl_refresh=args.enable_ssl_refresh,
        host=args.host,
        port=args.port,
        log_level=args.uvicorn_log_level,
        # NOTE: When the 'disable_uvicorn_access_log' value is True,
        # no access log will be output.
        access_log=not args.disable_uvicorn_access_log,
        timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_cert_reqs=args.ssl_cert_reqs,
        ssl_ciphers=args.ssl_ciphers,
        h11_max_incomplete_event_size=args.h11_max_incomplete_event_size,
        h11_max_header_count=args.h11_max_header_count,
        **uvicorn_kwargs,
    )

def build_app(
    args: Namespace,
    supported_tasks: tuple["SupportedTask", ...] | None = None,
    model_config: ModelConfig | None = None,
) -> FastAPI:

    if args.disable_fastapi_docs:
        app = FastAPI(
            openapi_url=None, docs_url=None, redoc_url=None, lifespan=lifespan
        )
    elif args.enable_offline_docs:
        app = FastAPI(docs_url=None, redoc_url=None, lifespan=lifespan)
    else:
        app = FastAPI(lifespan=lifespan)
    app.state.args = args
    app.root_path = args.root_path

	# 注册 api 路由
    register_api_routers(args, app, supported_tasks, model_config)

    # Endpoint plugins are attached last so their routes are registered after all core
    # routers. This runs even for the CPU only render server. A plugin eligible for
    # the `render` task still gets its routes registered. It receives
    # `engine_client=None` at Phase B (see `_init_endpoint_plugins_state`).
    attach_endpoint_plugins(app, supported_tasks)

    init_exception_handler(app)
    init_entrypoints_middleware(args, app, supported_tasks)
    app = sagemaker_standards_bootstrap(app)
    return app
```

