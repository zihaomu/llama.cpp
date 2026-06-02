#include <ggml-alloc.h>
#include <ggml-backend-impl.h>
#include <ggml-cpp.h>
#include <ggml-cpu.h>
#include <ggml.h>

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

struct offload_backend {
    ggml_backend backend;
    ggml_backend_device device;
    ggml_backend_buffer_type buffer_type;
};

static const char * offload_backend_name(ggml_backend_t) {
    return "OFFLOAD";
}

static void offload_backend_free(ggml_backend_t) {
}

static const char * offload_backend_buffer_type_get_name(ggml_backend_buffer_type_t) {
    return "OFFLOAD";
}

static void offload_backend_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    free(buffer->context);
}

static void * offload_backend_buffer_get_base(ggml_backend_buffer_t buffer) {
    return buffer->context;
}

static void offload_backend_buffer_memset_tensor(ggml_backend_buffer_t, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    memset((char *) tensor->data + offset, value, size);
}

static void offload_backend_buffer_set_tensor(ggml_backend_buffer_t, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    memcpy((char *) tensor->data + offset, data, size);
}

static void offload_backend_buffer_get_tensor(ggml_backend_buffer_t, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    memcpy(data, (const char *) tensor->data + offset, size);
}

static void offload_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    memset(buffer->context, value, buffer->size);
}

static const ggml_backend_buffer_i offload_backend_buffer_i = {
    /* .free_buffer     = */ offload_backend_buffer_free_buffer,
    /* .get_base        = */ offload_backend_buffer_get_base,
    /* .init_tensor     = */ nullptr,
    /* .memset_tensor   = */ offload_backend_buffer_memset_tensor,
    /* .set_tensor      = */ offload_backend_buffer_set_tensor,
    /* .get_tensor      = */ offload_backend_buffer_get_tensor,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ offload_backend_buffer_clear,
    /* .reset           = */ nullptr,
};

static ggml_backend_buffer_t offload_backend_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = malloc(size ? size : 1);
    GGML_ASSERT(data != nullptr);
    return ggml_backend_buffer_init(buft, offload_backend_buffer_i, data, size);
}

static size_t offload_backend_buffer_type_get_alignment(ggml_backend_buffer_type_t) {
    return ggml_backend_buft_get_alignment(ggml_backend_cpu_buffer_type());
}

static const char * offload_backend_device_get_name(ggml_backend_dev_t) {
    return "OFFLOAD0";
}

static const char * offload_backend_device_get_description(ggml_backend_dev_t) {
    return "test offload backend";
}

static void offload_backend_device_get_memory(ggml_backend_dev_t, size_t * free, size_t * total) {
    *free  = 0;
    *total = 0;
}

static enum ggml_backend_dev_type offload_backend_device_get_type(ggml_backend_dev_t) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void offload_backend_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = offload_backend_device_get_name(dev);
    props->description = offload_backend_device_get_description(dev);
    props->type        = offload_backend_device_get_type(dev);
}

static ggml_backend_t offload_backend_device_init_backend(ggml_backend_dev_t dev, const char *) {
    offload_backend * ctx = (offload_backend *) dev->context;
    return &ctx->backend;
}

static ggml_backend_buffer_type_t offload_backend_device_get_buffer_type(ggml_backend_dev_t dev) {
    offload_backend * ctx = (offload_backend *) dev->context;
    return &ctx->buffer_type;
}

static bool offload_backend_device_supports_op(ggml_backend_dev_t, const ggml_tensor * op) {
    return op->op == GGML_OP_ADD || op->op == GGML_OP_MUL;
}

static bool offload_backend_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    offload_backend * ctx = (offload_backend *) dev->context;
    return buft == &ctx->buffer_type;
}

static bool offload_backend_device_offload_op(ggml_backend_dev_t, const ggml_tensor * op) {
    return op->op == GGML_OP_ADD || op->op == GGML_OP_MUL;
}

static void init_offload_backend(offload_backend * ctx) {
    *ctx = {};

    ctx->buffer_type.iface.get_name      = offload_backend_buffer_type_get_name;
    ctx->buffer_type.iface.alloc_buffer  = offload_backend_buffer_type_alloc_buffer;
    ctx->buffer_type.iface.get_alignment = offload_backend_buffer_type_get_alignment;
    ctx->buffer_type.device              = &ctx->device;
    ctx->buffer_type.context             = ctx;

    ctx->device.iface.get_name        = offload_backend_device_get_name;
    ctx->device.iface.get_description = offload_backend_device_get_description;
    ctx->device.iface.get_memory      = offload_backend_device_get_memory;
    ctx->device.iface.get_type        = offload_backend_device_get_type;
    ctx->device.iface.get_props       = offload_backend_device_get_props;
    ctx->device.iface.init_backend    = offload_backend_device_init_backend;
    ctx->device.iface.get_buffer_type = offload_backend_device_get_buffer_type;
    ctx->device.iface.supports_op     = offload_backend_device_supports_op;
    ctx->device.iface.supports_buft   = offload_backend_device_supports_buft;
    ctx->device.iface.offload_op      = offload_backend_device_offload_op;
    ctx->device.context               = ctx;

    ctx->backend.iface.get_name = offload_backend_name;
    ctx->backend.iface.free     = offload_backend_free;
    ctx->backend.device         = &ctx->device;
    ctx->backend.context        = ctx;
}

static ggml_context_ptr make_context(size_t extra_tensors = 0) {
    ggml_init_params params = {
        /* .mem_size   = */ (8 + extra_tensors)*ggml_tensor_overhead() + ggml_graph_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };

    return ggml_context_ptr(ggml_init(params));
}

static void set_env_var(const char * name, const char * value) {
#if defined(_WIN32)
    _putenv_s(name, value != nullptr ? value : "");
#else
    if (value != nullptr) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

struct scoped_env_var {
    const char * name;
    std::string old_value;
    bool had_old_value;

    scoped_env_var(const char * name, const char * value) : name(name), had_old_value(getenv(name) != nullptr) {
        if (had_old_value) {
            old_value = getenv(name);
        }

        set_env_var(name, value);
    }

    ~scoped_env_var() {
        if (had_old_value) {
            set_env_var(name, old_value.c_str());
        } else {
            set_env_var(name, nullptr);
        }
    }
};

static ggml_backend_t assign_host_weight_add(ggml_backend_t cpu_backend, offload_backend * offload, bool use_weight_view) {
    ggml_context_ptr ctx_weights = make_context();
    ggml_tensor * weight = ggml_new_tensor_1d(ctx_weights.get(), GGML_TYPE_F32, 16);
    ggml_set_name(weight, "host_weight");

    ggml_backend_buffer_ptr weights(ggml_backend_alloc_ctx_tensors_from_buft(ctx_weights.get(), ggml_backend_cpu_buffer_type()));
    GGML_ASSERT(weights != nullptr);
    ggml_backend_buffer_set_usage(weights.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    ggml_context_ptr ctx = make_context();
    ggml_tensor * input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 16);
    ggml_set_input(input);
    ggml_set_name(input, "input");

    ggml_tensor * add_weight = weight;
    if (use_weight_view) {
        add_weight = ggml_view_1d(ctx.get(), weight, 16, 0);
        ggml_set_name(add_weight, "host_weight_view");
    }

    ggml_tensor * out = ggml_add(ctx.get(), input, add_weight);
    ggml_set_name(out, "out");
    ggml_set_output(out);

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);

    ggml_backend_t backends[] = { &offload->backend, cpu_backend };
    ggml_backend_buffer_type_t bufts[] = { &offload->buffer_type, ggml_backend_cpu_buffer_type() };
    ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, bufts, 2, GGML_DEFAULT_GRAPH_SIZE, false, true));

    ggml_backend_sched_split_graph(sched.get(), graph);

    return ggml_backend_sched_get_tensor_backend(sched.get(), out);
}

static void test_host_weight_op_offload_guard() {
    scoped_env_var env("GGML_OP_OFFLOAD_MAX_HOST_WEIGHT_BYTES", nullptr);
    offload_backend offload;
    init_offload_backend(&offload);
    ggml_backend_ptr cpu(ggml_backend_cpu_init());

    GGML_ASSERT(assign_host_weight_add(cpu.get(), &offload, false) == cpu.get());
    GGML_ASSERT(assign_host_weight_add(cpu.get(), &offload, true) == cpu.get());
}

static void test_host_weight_op_offload_env_override() {
    scoped_env_var env("GGML_OP_OFFLOAD_MAX_HOST_WEIGHT_BYTES", "1024");
    offload_backend offload;
    init_offload_backend(&offload);
    ggml_backend_ptr cpu(ggml_backend_cpu_init());

    GGML_ASSERT(assign_host_weight_add(cpu.get(), &offload, false) == &offload.backend);
    GGML_ASSERT(assign_host_weight_add(cpu.get(), &offload, true) == &offload.backend);
}

static void run(const char * name, void (*fn)()) {
    printf("%s ", name);
    fflush(stdout);
    fn();
    printf("PASSED\n");
}

int main() {
    run("test_host_weight_op_offload_guard", test_host_weight_op_offload_guard);
    run("test_host_weight_op_offload_env_override", test_host_weight_op_offload_env_override);
    return 0;
}
