# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""BUILD extensions for MLIR table generation."""

load("@bazel_skylib//lib:paths.bzl", "paths")

TdInfo = provider(
    "Holds TableGen files and the dependencies and include paths necessary to" +
    " build them.",
    fields = {
        "transitive_sources": "td files transitively used by this rule.",
        "transitive_includes": (
            "include arguments to add to the final TableGen invocation. These" +
            " are the absolute directory paths that will be added with '-I'."
        ),
    },
)

# For now we allow anything that provides DefaultInfo to just forward its files.
# In particular, this allows filegroups to be used. This is mostly to ease
# transition. In the future, the TdInfo provider will be required.
# TODO(gcmn): Switch to enforcing TdInfo provider.
def _get_dep_transitive_srcs(dep):
    """Extract TdInfo.transitive_sources, falling back to DefaultInfo.files."""
    if TdInfo in dep:
        return dep[TdInfo].transitive_sources
    return dep[DefaultInfo].files

def _get_dep_transitive_includes(dep):
    """Extract TdInfo.transitive_includes, falling back to an empty depset()."""
    if TdInfo in dep:
        return dep[TdInfo].transitive_includes
    return depset()

def _get_transitive_srcs(srcs, deps):
    """Obtain the source files for a target and its transitive dependencies.

    Args:
      srcs: a list of source files
      deps: a list of targets that are direct dependencies
    Returns:
      a collection of the transitive sources
    """
    return depset(
        direct = srcs,
        transitive = [_get_dep_transitive_srcs(dep) for dep in deps],
    )

def _get_transitive_includes(includes, deps):
    """Obtain the includes paths for a target and its transitive dependencies.

    Args:
      includes: a list of include paths
      deps: a list of targets that are direct dependencies
    Returns:
      a collection of the transitive include paths
    """
    return depset(
        direct = includes,
        transitive = [_get_dep_transitive_includes(dep) for dep in deps],
    )

def _prefix_roots(ctx, includes):
    """Map the given includes to be relative to all root directories.

    This will expand them to be relative to all the root directories available
    in the execution environment for ctx.run (bin and genfiles in addition to
    the normal source root)
    """
    prefixed_includes = []
    for include in includes:
        prefixed_includes.append(include)
        prefixed_includes.append(paths.join(ctx.genfiles_dir.path, include))
        prefixed_includes.append(paths.join(ctx.bin_dir.path, include))
    return prefixed_includes

def _resolve_includes(ctx, includes):
    """Resolves include paths to paths relative to the execution root.

    Relative paths are interpreted as relative to the current label's package.
    Absolute paths are interpreted as relative to the current label's workspace
    root."""
    package = ctx.label.package
    workspace_root = ctx.label.workspace_root
    workspace_root = workspace_root if workspace_root else "."
    resolved_includes = []
    for include in includes:
        if paths.is_absolute(include):
            include = include.lstrip("/")
        else:
            include = paths.join(package, include)
        include = paths.join(workspace_root, include)
        resolved_includes.extend(_prefix_roots(ctx, [include]))
    return resolved_includes

def _td_library_impl(ctx):
    trans_srcs = _get_transitive_srcs(ctx.files.srcs, ctx.attr.deps)
    trans_includes = _get_transitive_includes(
        _resolve_includes(ctx, ctx.attr.includes),
        ctx.attr.deps,
    )

    # Note that we include srcs in runfiles. A td_library doesn't compile to
    # produce an output: it's just a depset of source files and include
    # directories. So if it is needed for execution of some rule (likely
    # something running tblgen as a test action), the files needed are the same
    # as the source files.
    # Note: not using merge_all, as that is not available in Bazel 4.0
    runfiles = ctx.runfiles(ctx.files.srcs)
    for src in ctx.attr.srcs:
        runfiles = runfiles.merge(src[DefaultInfo].default_runfiles)
    for dep in ctx.attr.deps:
        runfiles = runfiles.merge(dep[DefaultInfo].default_runfiles)

    return [
        DefaultInfo(files = trans_srcs, runfiles = runfiles),
        TdInfo(
            transitive_sources = trans_srcs,
            transitive_includes = trans_includes,
        ),
    ]

td_library = rule(
    _td_library_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = True),
        "includes": attr.string_list(
            doc = "Include paths to be added to the final TableGen tool" +
                  " invocation. Relative paths are interpreted as relative to" +
                  " the current label's package. Absolute paths are" +
                  " interpreted as relative to the current label's workspace",
        ),
        # TODO(gcmn): limit to TdInfo providers.
        "deps": attr.label_list(
            doc = "Dependencies providing TableGen source files and include" +
                  " paths.",
        ),
    },
)

def _gentbl_rule_impl(ctx):
    td_file = ctx.file.td_file

    trans_srcs = _get_transitive_srcs(
        ctx.files.td_srcs + [td_file],
        ctx.attr.deps,
    )

    # Note that the td_file.dirname is already relative to the execution root,
    # i.e. may contain an `external/<workspace_name>` prefix if the current
    # workspace is not the main workspace. Therefore it is not included in the
    # _resolve_includes call that prepends this prefix.
    trans_includes = _get_transitive_includes(
        _resolve_includes(ctx, ctx.attr.includes + ["/"]) +
        _prefix_roots(ctx, [td_file.dirname]),
        ctx.attr.deps,
    )

    args = ctx.actions.args()
    args.add_all(ctx.attr.opts)
    args.add(td_file)
    args.add_all(trans_includes, before_each = "-I")

    args.add("-o", ctx.outputs.out.path)

    ctx.actions.run(
        outputs = [ctx.outputs.out],
        inputs = trans_srcs,
        executable = ctx.executable.tblgen,
        arguments = [args],
        # Make sure action_env settings are honored so the env is the same as
        # when the tool was built. Important for locating shared libraries with
        # a custom LD_LIBRARY_PATH.
        use_default_shell_env = True,
        mnemonic = "TdGenerate",
    )

    return [DefaultInfo()]

gentbl_rule = rule(
    _gentbl_rule_impl,
    doc = "Generates tabular code from a table definition file.",
    # Match genrule behavior
    output_to_genfiles = True,
    attrs = {
        "tblgen": attr.label(
            doc = "The TableGen executable with which to generate `out`.",
            executable = True,
            cfg = "exec",
        ),
        "td_file": attr.label(
            doc = "The TableGen file to run through `tblgen`.",
            allow_single_file = True,
            mandatory = True,
        ),
        "td_srcs": attr.label_list(
            doc = "Additional TableGen files included by `td_file`. It is not" +
                  " necessary to list td_file here (though not an error).",
            allow_files = True,
        ),
        # TODO(gcmn): limit to TdInfo providers.
        "deps": attr.label_list(
            doc = "Dependencies providing TableGen source files and include" +
                  " paths.",
        ),
        "out": attr.output(
            doc = "The output file for the TableGen invocation.",
            mandatory = True,
        ),
        "opts": attr.string_list(
            doc = "Additional command line options to add to the TableGen" +
                  " invocation. For include arguments, prefer to use" +
                  " `includes`.",
        ),
        "includes": attr.string_list(
            doc = "Include paths to be added to the final TableGen tool" +
                  " invocation. Relative paths are interpreted as relative to" +
                  " the current label's package. Absolute paths are" +
                  " interpreted as relative to the current label's workspace." +
                  " Includes are applied from all roots available in the" +
                  " execution environment (source, genfiles, and bin" +
                  " directories). The execution roots themselves and the " +
                  " directory of td_file are always added.",
        ),
    },
)

# TODO(gcmn): Figure out how to reduce duplication with _gentbl_rule_impl
def _gentbl_test_impl(ctx):
    td_file = ctx.file.td_file

    # Note that the td_file.dirname is already relative to the execution root,
    # i.e. may contain an `external/<workspace_name>` prefix if the current
    # workspace is not the main workspace. Therefore it is not included in the
    # _resolve_includes call that prepends this prefix.
    trans_includes = _get_transitive_includes(
        _resolve_includes(ctx, ctx.attr.includes + ["/"]) +
        _prefix_roots(ctx, [td_file.dirname]),
        ctx.attr.deps,
    )

    test_args = [ctx.executable.tblgen.short_path]
    test_args.extend(ctx.attr.opts)
    test_args.append(td_file.path)
    test_args.extend(["-I " + include for include in trans_includes.to_list()])

    test_args.extend(["-o", "/dev/null"])

    ctx.actions.write(
        ctx.outputs.executable,
        content = " ".join(test_args),
        is_executable = True,
    )

    # Note: not using merge_all, as that is not available in Bazel 4.0
    runfiles = ctx.runfiles(
        files = [ctx.executable.tblgen],
        transitive_files = _get_transitive_srcs(
            ctx.files.td_srcs + [td_file],
            ctx.attr.deps,
        ),
    )
    for src in ctx.attr.td_srcs:
        runfiles = runfiles.merge(src[DefaultInfo].default_runfiles)
    for dep in ctx.attr.deps:
        runfiles = runfiles.merge(dep[DefaultInfo].default_runfiles)

    return [
        coverage_common.instrumented_files_info(
            ctx,
            source_attributes = ["td_file", "td_srcs"],
            dependency_attributes = ["tblgen", "deps"],
        ),
        DefaultInfo(runfiles = runfiles),
    ]

gentbl_test = rule(
    _gentbl_test_impl,
    test = True,
    doc = "A shell test that tests the given TablegGen invocation. Note" +
          " that unlike gentbl_rule, this builds and invokes `tblgen` in the" +
          " target configuration. Takes all the same arguments as gentbl_rule" +
          " except for `out` (as it does not generate any output)",
    attrs = {
        "tblgen": attr.label(
            doc = "The TableGen executable run in the shell command. Note" +
                  " that this is built in the target configuration.",
            executable = True,
            cfg = "target",
        ),
        "td_file": attr.label(
            doc = "See gentbl_rule.td_file",
            allow_single_file = True,
            mandatory = True,
        ),
        "td_srcs": attr.label_list(
            doc = "See gentbl_rule.td_srcs",
            allow_files = True,
        ),
        "deps": attr.label_list(doc = "See gentbl_rule.deps"),
        "opts": attr.string_list(doc = "See gentbl_rule.opts"),
        "includes": attr.string_list(doc = "See gentbl_rule.includes"),
    },
)

def gentbl_filegroup(
        name,
        tblgen,
        td_file,
        tbl_outs,
        td_srcs = [],
        includes = [],
        deps = [],
        test = False,
        skip_opts = [],
        **kwargs):
    """Create multiple TableGen generated files using the same tool and input.

    All generated outputs are bundled in a file group with the given name.

    Args:
      name: The name of the generated filegroup rule for use in dependencies.
      tblgen: The binary used to produce the output.
      td_file: The primary table definitions file.
      tbl_outs: A list of tuples ([opts], out), where each 'opts' is a list of
        options passed to tblgen, each option being a string, and 'out' is the
        corresponding output file produced.
      td_srcs: See gentbl_rule.td_srcs
      includes: See gentbl_rule.includes
      deps: See gentbl_rule.deps
      test: Whether to create a shell test that invokes the tool too.
      skip_opts: Files generated using these opts in tbl_outs will be excluded
        from the generated filegroup.
      **kwargs: Extra keyword arguments to pass to all generated rules.
    """

    for (opts, out) in tbl_outs:
        first_opt = opts[0] if opts else ""
        rule_suffix = "_{}_{}".format(
            first_opt.replace("-", "_").replace("=", "_"),
            str(hash(" ".join(opts))),
        )
        gentbl_name = "%s_%s_genrule" % (name, rule_suffix)
        gentbl_rule(
            name = gentbl_name,
            td_file = td_file,
            tblgen = tblgen,
            opts = opts,
            td_srcs = td_srcs,
            deps = deps,
            includes = includes,
            out = out,
            **kwargs
        )

        if test:
            # Also run the generator in the target configuration as a test. This
            # means it gets run with asserts and sanitizers and such when they
            # are enabled and is counted in coverage.
            gentbl_test(
                name = "%s_test" % (gentbl_name,),
                td_file = td_file,
                tblgen = tblgen,
                opts = opts,
                td_srcs = td_srcs,
                deps = deps,
                includes = includes,
                # Shell files not executable on Windows.
                # TODO(gcmn): Support windows.
                tags = ["no_windows"],
                **kwargs
            )

    included_srcs = [f for (opts, f) in tbl_outs if not any([skip_opt in opts for skip_opt in skip_opts])]
    native.filegroup(
        name = name,
        srcs = included_srcs,
        **kwargs
    )

def gentbl_cc_library(
        name,
        tblgen,
        td_file,
        tbl_outs,
        td_srcs = [],
        includes = [],
        deps = [],
        strip_include_prefix = None,
        test = False,
        **kwargs):
    """Create multiple TableGen generated files using the same tool and input.

    All generated outputs are bundled in a cc_library rule.

    Args:
      name: The name of the generated cc_library rule for use in dependencies.
      tblgen: The binary used to produce the output.
      td_file: The primary table definitions file.
      tbl_outs: A list of tuples ([opts], out), where each 'opts' is a list of
        options passed to tblgen, each option being a string, and 'out' is the
        corresponding output file produced.
      td_srcs: See gentbl_rule.td_srcs
      includes: See gentbl_rule.includes
      deps: See gentbl_rule.deps
      strip_include_prefix: attribute to pass through to cc_library.
      test: whether to create a shell test that invokes the tool too.
      **kwargs: Extra keyword arguments to pass to all generated rules.
    """

    filegroup_name = name + "_filegroup"
    gentbl_filegroup(
        name = filegroup_name,
        tblgen = tblgen,
        td_file = td_file,
        tbl_outs = tbl_outs,
        td_srcs = td_srcs,
        includes = includes,
        deps = deps,
        test = test,
        skip_opts = ["-gen-op-doc"],
        **kwargs
    )
    native.cc_library(
        name = name,
        # strip_include_prefix does not apply to textual_hdrs.
        # https://github.com/bazelbuild/bazel/issues/12424
        hdrs = [":" + filegroup_name] if strip_include_prefix else [],
        strip_include_prefix = strip_include_prefix,
        textual_hdrs = [":" + filegroup_name],
        **kwargs
    )
