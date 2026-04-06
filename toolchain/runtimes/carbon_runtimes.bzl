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

def _build_crt_file(ctx, cc_toolchain, feature_configuration, crt_file):
    _, compilation_outputs = cc_common.compile(
        name = ctx.label.name + ".compile_" + crt_file.basename,
        actions = ctx.actions,
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        srcs = [crt_file],
        user_compile_flags = ctx.attr.crt_copts,
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

def _carbon_runtimes_impl(ctx):
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

    if ctx.attr.target_triple != "":
        builtins_lib_path = "clang_resource_dir/lib/{0}".format(ctx.attr.target_triple)
    elif ctx.attr.darwin_os_suffix:
        builtins_lib_path = "clang_resource_dir/lib/darwin"
        builtins_archive_name = "libclang_rt.{0}.a".format(ctx.attr.darwin_os_suffix)

    for filename, src in [
        ("crtbegin", ctx.files.crtbegin_src),
        ("crtend", ctx.files.crtend_src),
    ]:
        if not src:
            continue
        src = src[0]
        crt_obj = _build_crt_file(ctx, cc_toolchain, feature_configuration, src)
        crt_out = ctx.actions.declare_file("{0}/{1}/clang_rt.{2}.o".format(
            prefix,
            builtins_lib_path,
            filename,
        ))
        ctx.actions.symlink(output = crt_out, target_file = crt_obj)
        outputs.append(crt_out)

    for runtime_dir, archive_name, archive in [
        (builtins_lib_path, builtins_archive_name, ctx.files.builtins_archive[0]),
        ("libcxx/lib", "libc++.a", ctx.files.libcxx_archive[0]),
        ("libunwind/lib", "libunwind.a", ctx.files.libunwind_archive[0]),
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
            rel_path = _removeprefix_or_fail(rel_path, hdr.root.path + "/")
        if hdr.owner.workspace_root != "":
            rel_path = _removeprefix_or_fail(rel_path, hdr.owner.workspace_root + "/")
        if hdr.owner.package != "":
            rel_path = _removeprefix_or_fail(rel_path, hdr.owner.package + "/")
        rel_path = _removeprefix_or_fail(rel_path, ctx.attr.clang_hdrs_prefix)

        out_hdr = ctx.actions.declare_file(
            "{0}/clang_resource_dir/include/{1}".format(prefix, rel_path),
        )
        ctx.actions.symlink(output = out_hdr, target_file = hdr)
        outputs.append(out_hdr)

    return [DefaultInfo(files = depset(outputs))]

carbon_runtimes = rule(
    implementation = _carbon_runtimes_impl,
    attrs = {
        "builtins_archive": attr.label(mandatory = True, allow_files = [".a"]),
        "clang_hdrs": attr.label_list(mandatory = True, allow_files = True),
        "clang_hdrs_prefix": attr.string(default = "include/"),
        "crt_copts": attr.string_list(default = []),
        "crtbegin_src": attr.label(allow_files = [".c"]),
        "crtend_src": attr.label(allow_files = [".c"]),
        "darwin_os_suffix": attr.string(mandatory = False),
        "libcxx_archive": attr.label(mandatory = True, allow_files = [".a"]),
        "libunwind_archive": attr.label(mandatory = True, allow_files = [".a"]),
        "target_triple": attr.string(mandatory = False),
        "_cc_toolchain": attr.label(
            default = Label("@bazel_tools//tools/cpp:current_cc_toolchain"),
        ),
    },
    toolchains = ["@bazel_tools//tools/cpp:toolchain_type"],
    fragments = ["cpp"],
    doc = """Builds a Carbon runtimes tree under the `name` directory.

    This rule works to replicate the behavior of the Carbon toolchain building a
    runtimes tree directly in Bazel to allow it to fully benefit from Bazel's
    orchestration, caching, and even remote execution.

    It produces a runtimes tree customized for the specific target platform,
    similar to what the Carbon toolchain does on its own when invoked outside of
    Bazel.

    The runtimes tree includes a complete Clang resource-dir, including CRT
    begin/end objects, and the builtins library. It also includes a built libc++
    and libunwind library. These are arranged in the standard Carbon runtimes
    layout so that they are correctly found by the toolchain.
    """,
)
