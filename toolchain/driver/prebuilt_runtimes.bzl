# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rules for pre-building a runtimes tree."""

def _prebuilt_clang_runtimes_impl(ctx):
    runtimes_builder = ctx.executable.internal_exec_runtimes_builder
    if runtimes_builder == None:
        runtimes_builder = ctx.executable.internal_target_runtimes_builder

    # Declare the output directories created when building the runtimes. We
    # can't declare just the top-level directory as we need to overlay an
    # `include` tree into the `clang_resource_dir`, but we try to use
    # directories to minimize writing out the complete list of files in each
    # runtime.
    output_dirs = [
        ctx.actions.declare_directory(ctx.attr.runtimes_path + "/clang_resource_dir/lib"),
        ctx.actions.declare_directory(ctx.attr.runtimes_path + "/libunwind/lib"),
        ctx.actions.declare_directory(ctx.attr.runtimes_path + "/libcxx/lib"),
    ]

    # Build up the arguments to use with the runtimes build command.
    args = ctx.actions.args()
    args.add("--force")

    # Provide an explicit target if requested.
    if ctx.attr.target:
        args.add(ctx.attr.target, format = "--target=%s")

    # Compute the path to the root of the runtimes tree we're trying to build.
    # We work backwards from the declared `clang_resource_dir/lib` entry.
    root_path_arg = output_dirs[0].path
    if not root_path_arg.endswith("/clang_resource_dir/lib"):
        fail("Unexpected path structure: " + root_path_arg)
    root_path_arg = root_path_arg[:-len("/clang_resource_dir/lib")]
    args.add(root_path_arg)

    # Run the runtimes building tool with the arguments.
    ctx.actions.run(
        outputs = output_dirs,
        executable = runtimes_builder,
        arguments = [args],
        mnemonic = "BuildRuntimes",
        progress_message = "Building runtimes target %{label}",

        # Building runtimes will use all the available CPUs. We can't directly
        # model this in Bazel, so we use a somewhat arbitrary but large number
        # of cores here to indicate that this is a _very_ expensive action. This
        # should minimize the risk of other actions running in parallel in
        # constrained environments and timing out.
        execution_requirements = {"cpu:64": ""},
    )

    # Now overlay Clang's builtin headers to the `clang_resource_dir` as we
    # can't have separate library search and header search paths for that
    # specific runtime, and we want Bazel to be aware of the origin of these
    # headers rather than creating copies when building the runtimes.
    target_include_root = ctx.attr.runtimes_path + "/clang_resource_dir/include"

    # We need to compute the relative path of each header file within the
    # resource directory's `include` directory. We do this by stripping off
    # their prefix.
    #
    # TODO: It would be nice to find a cleaner way to do this that avoids
    # hard-coding both the repository name's spelling and the rule layout.
    headers_prefix = "/external/+llvm_project+llvm-project/clang/staging/include/"

    # Walk all the headers and symlink them into the runtimes tree below the
    # target root. Collect the results for use in establishing dependencies.
    input_headers = ctx.attr._builtin_headers.files.to_list()
    output_headers = []
    for f in input_headers:
        # Ensure the file actually lives under the expected path
        path = f.path[len(f.root.path):]
        if not path.startswith(headers_prefix):
            fail("Header file '{}' is not under the expected prefix '{}'".format(
                path,
                headers_prefix,
            ))

        path = path[len(headers_prefix):]

        # Declare the output file preserving the relative structure. Bazel
        # automatically creates any intermediate directories (e.g. `sanitizer/`)
        out_file = ctx.actions.declare_file("{}/{}".format(target_include_root, path))

        ctx.actions.symlink(output = out_file, target_file = f)
        output_headers.append(out_file)

    return [
        DefaultInfo(
            # Build the actual runtimes tree when the target is built.
            files = depset(output_dirs + output_headers),
            # Make the resulting tree available in the Bazel runfiles tree.
            runfiles = ctx.runfiles(files = output_dirs + output_headers),
        ),
    ]

# The rule implementing the Clang runtimes build. This is kept in its own rule
# and uses separate tools so that it can isolate its dependencies and thus how
# often it has to be re-built as much as possible.
_prebuilt_clang_runtimes_internal = rule(
    implementation = _prebuilt_clang_runtimes_impl,
    attrs = {
        # These are technically builtin attributes, but we provide them via a
        # macro in order to use select to configure them based on a flag.
        "internal_exec_runtimes_builder": attr.label(
            allow_single_file = True,
            executable = True,
            cfg = "exec",
        ),
        "internal_target_runtimes_builder": attr.label(
            allow_single_file = True,
            executable = True,
            cfg = "target",
        ),
        "runtimes_path": attr.string(
            doc = "The path for the root of the runtimes",
            mandatory = True,
        ),
        "target": attr.string(
            doc = "Optional target for the built runtimes",
        ),
        "_builtin_headers": attr.label(
            default = Label("//toolchain/install:clang_headers"),
        ),
    },
)

def prebuilt_runtimes(name, target = None, tags = []):
    """Build a a runtimes tree.

    The runtimes will be built into the directory `name + "_tree"`, and
    collected into a filegroup with the provided name for use in rules accessing
    these runtimes.

    Args:
      name: The name of the runtimes build target.
      target: Optional `--target` flag value to use when building the runtimes.
      tags: Tags to apply to the rule.
    """
    runtimes_path = name + "_tree"

    _prebuilt_clang_runtimes_internal(
        name = name + "_clang",
        runtimes_path = runtimes_path,
        target = target,
        tags = tags,

        # Synthesize mirrored `select`-filled attributes here so that they can
        # have different internal properties (that can't be `select`-ed) and we
        # can select between the attributes instead.
        internal_exec_runtimes_builder = select({
            "//toolchain/driver:use_target_config_runtimes_builder_config": None,
            "//conditions:default": "//toolchain/driver:bazel_build_clang_runtimes",
        }),
        internal_target_runtimes_builder = select({
            "//toolchain/driver:use_target_config_runtimes_builder_config": "//toolchain/driver:bazel_build_clang_runtimes",
            "//conditions:default": None,
        }),
    )

    # TODO: Add building of the Carbon runtimes here using a parallel rule to
    # the above, but adjusted to use the Carbon driver itself as there will be
    # no dependency reduction to be gained with a dedicated tool. It should
    # reuse `runtimes_path` so that we get a unified runtimes tree for
    # downstream use.

    # Assemble the various runtimes into a single filegroup for easy
    # dependencies.
    native.filegroup(
        name = name,
        srcs = [
            ":" + name + "_clang",
        ],
    )
