# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Starlark rules for building a Carbon runtimes tree.

TODO: Currently, this produces a complete, static Carbon runtimes tree that
mirrors the exact style of runtimes tree the Carbon toolchain would build on its
own. However, it would be preferable to preserve the builtins, libc++, and
libunwind `cc_library` rules as "normal" library rules (if behind a transition)
and automatically depend on them. This would allow things like LTO and such to
include these. However, this requires support in `@rules_cc` for this kind of
dependency to be added.
"""

load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain")
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")

def _build_crt_file(ctx, cc_toolchain, feature_configuration, crt_file, crt_copts):
    _, compilation_outputs = cc_common.compile(
        name = "{}.compile_{}".format(ctx.label.name, crt_file.basename),
        actions = ctx.actions,
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        srcs = [crt_file],
        user_compile_flags = crt_copts,
    )

    # Extract the PIC object file and make sure we built one.
    obj = compilation_outputs.pic_objects[0]
    if not obj:
        fail("The toolchain failed to produce a PIC object file. Ensure your " +
             "toolchain supports PIC.")

    return obj

def _removeprefix_or_fail(s, prefix):
    new_s = s.removeprefix(prefix)
    if new_s == s:
        fail("Unable to remove prefix '{0}' from '{1}'".format(prefix, s))
    return new_s

CarbonRuntimesConfigInfo = provider(
    doc = """Configuration for Carbon runtimes.

    This provider is used to collect all of the information that will be needed
    to build a Carbon runtimes directory for a Bazel Carbon `cc_toolchain`.
    """,
    fields = [
        "builtins_archive",
        "clang_hdrs_prefix",
        "crt_copts",
        "crtbegin_src",
        "crtend_src",
        "darwin_os_suffix",
        "libcxx_archive",
        "libunwind_archive",
        "target_triple",
    ],
)

def _carbon_runtimes_config_impl(ctx):
    return [
        CarbonRuntimesConfigInfo(
            builtins_archive = ctx.files.builtins_archive[0],
            clang_hdrs_prefix = ctx.attr.clang_hdrs_prefix,
            crt_copts = ctx.attr.crt_copts,
            crtbegin_src = ctx.files.crtbegin_src[0] if ctx.files.crtbegin_src else None,
            crtend_src = ctx.files.crtend_src[0] if ctx.files.crtend_src else None,
            darwin_os_suffix = ctx.attr.darwin_os_suffix,
            libcxx_archive = ctx.files.libcxx_archive[0],
            libunwind_archive = ctx.files.libunwind_archive[0],
            target_triple = ctx.attr.target_triple,
        ),
    ]

carbon_runtimes_config = rule(
    implementation = _carbon_runtimes_config_impl,
    attrs = {
        "builtins_archive": attr.label(mandatory = True, allow_files = [".a"]),
        "clang_hdrs_prefix": attr.string(default = "include/"),
        "crt_copts": attr.string_list(default = []),
        "crtbegin_src": attr.label(allow_files = [".c"]),
        "crtend_src": attr.label(allow_files = [".c"]),
        "darwin_os_suffix": attr.string(mandatory = False),
        "libcxx_archive": attr.label(mandatory = True, allow_files = [".a"]),
        "libunwind_archive": attr.label(mandatory = True, allow_files = [".a"]),
        "target_triple": attr.string(mandatory = False),
    },
    doc = "Collects configuration for building a Carbon runtimes tree.",
)

def _carbon_runtimes_build_impl(ctx):
    config = ctx.attr.config[CarbonRuntimesConfigInfo]
    outputs = []
    prefix = ctx.attr.name

    # Create a marker file in the runtimes root first. We'll use this to locate
    # the runtimes for the toolchain.
    root_out = ctx.actions.declare_file("{0}/runtimes_root".format(prefix))
    ctx.actions.write(output = root_out, content = "")
    outputs.append(root_out)

    # Setup the C++ toolchain and configuration. We also force the `pic` feature
    # to be enabled for these actions as we always want PIC generated code --
    # this avoids the need to build two versions of the runtimes and doesn't
    # create problems with modern code generation when linking statically. This
    # also simplifies extracting the outputs as we only need to look at
    # `pic_objects`.
    cc_toolchain = find_cpp_toolchain(ctx)
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features + ["pic"],
        unsupported_features = ctx.disabled_features,
    )

    builtins_lib_path = "clang_resource_dir/lib"
    builtins_archive_name = "libclang_rt.builtins.a"

    if config.target_triple != "":
        builtins_lib_path = "clang_resource_dir/lib/{0}".format(config.target_triple)
    elif config.darwin_os_suffix:
        builtins_lib_path = "clang_resource_dir/lib/darwin"
        builtins_archive_name = "libclang_rt.{0}.a".format(config.darwin_os_suffix)

    for filename, src in [
        ("crtbegin", config.crtbegin_src),
        ("crtend", config.crtend_src),
    ]:
        if not src:
            continue
        crt_obj = _build_crt_file(ctx, cc_toolchain, feature_configuration, src, config.crt_copts)
        crt_out = ctx.actions.declare_file("{0}/{1}/clang_rt.{2}.o".format(
            prefix,
            builtins_lib_path,
            filename,
        ))
        ctx.actions.symlink(output = crt_out, target_file = crt_obj)
        outputs.append(crt_out)

    for runtime_dir, archive_name, archive in [
        (builtins_lib_path, builtins_archive_name, config.builtins_archive),
        ("libcxx/lib", "libc++.a", config.libcxx_archive),
        ("libunwind/lib", "libunwind.a", config.libunwind_archive),
    ]:
        runtime_out = ctx.actions.declare_file("{0}/{1}/{2}".format(
            prefix,
            runtime_dir,
            archive_name,
        ))
        ctx.actions.symlink(output = runtime_out, target_file = archive)
        outputs.append(runtime_out)

    for hdr in ctx.files.clang_hdrs:
        # Incrementally remove prefixes of the paths to find the `include`
        # directory we want to symlink into the output tree.
        rel_path = hdr.path
        if hdr.root.path != "":
            rel_path = _removeprefix_or_fail(rel_path, "{}/".format(hdr.root.path))
        if hdr.owner.workspace_root != "":
            rel_path = _removeprefix_or_fail(rel_path, "{}/".format(hdr.owner.workspace_root))
        if hdr.owner.package != "":
            rel_path = _removeprefix_or_fail(rel_path, "{}/".format(hdr.owner.package))
        rel_path = _removeprefix_or_fail(rel_path, config.clang_hdrs_prefix)

        out_hdr = ctx.actions.declare_file(
            "{0}/clang_resource_dir/include/{1}".format(prefix, rel_path),
        )
        ctx.actions.symlink(output = out_hdr, target_file = hdr)
        outputs.append(out_hdr)

    return [DefaultInfo(files = depset(outputs))]

carbon_runtimes_build = rule(
    implementation = _carbon_runtimes_build_impl,
    attrs = {
        "clang_hdrs": attr.label_list(
            mandatory = True,
            allow_files = True,
        ),
        "config": attr.label(mandatory = True, providers = [CarbonRuntimesConfigInfo]),
        "_cc_toolchain": attr.label(
            default = Label("@bazel_tools//tools/cpp:current_cc_toolchain"),
        ),
    },
    toolchains = ["@bazel_tools//tools/cpp:toolchain_type"],
    fragments = ["cpp"],
    doc = """Builds a Carbon runtimes tree using a config rule and clang_hdrs.

    The configuration provides access to all of the targets that should be built
    into the runtimes.

    Any files, such as `clang_hdrs`, that should be built _prior_ to the
    runtimes build taking place are accepted as separate file groups so that
    they can be properly handled when bootstrapping.
    """,
)
