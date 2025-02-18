# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rules for constructing install information."""

load("@rules_pkg//pkg:mappings.bzl", "pkg_attributes", "pkg_filegroup", "pkg_files", "pkg_mklink", "strip_prefix")
load("symlink_helpers.bzl", "symlink_file", "symlink_filegroup")

def install_filegroup(name, filegroup_target, remove_prefix = ""):
    """Adds a filegroup for install.

    Used in the `install_dirs` dict.

    Args:
      name: The base directory for the filegroup.
      filegroup_target: The bazel filegroup target to install.
    """
    return {
        "filegroup": filegroup_target,
        "is_driver": False,
        "name": name,
        "remove_prefix": remove_prefix,
    }

def install_symlink(name, symlink_to, is_driver = False):
    """Adds a symlink for install.

    Used in the `install_dirs` dict.

    Args:
      name: The filename to use.
      symlink_to: A relative path for the symlink.
      is_driver: False if it should be included in the `no_driver_name`
        filegroup.
    """
    return {
        "is_driver": is_driver,
        "name": name,
        "symlink": symlink_to,
    }

def install_target(name, target, executable = False, is_driver = False):
    """Adds a target for install.

    Used in the `install_dirs` dict.

    Args:
      name: The filename to use.
      target: The bazel target being installed.
      executable: True if executable.
      is_driver: False if it should be included in the `no_driver_name`
        filegroup.
    """
    return {
        "executable": executable,
        "is_driver": is_driver,
        "name": name,
        "target": target,
    }

def make_install_filegroups(name, no_driver_name, pkg_name, install_dirs, prefix):
    """Makes filegroups of install data.

    Args:
      name: The name of the main filegroup, that contains all install_data.
      no_driver_name: The name of a filegroup which excludes the driver. This is
        for the driver to depend on and get other files, without a circular
        dependency.
      pkg_name: The name of a pkg_filegroup for tar.
      install_dirs: A dict of {directory: [install_* rules]}. This is used to
        structure files to be installed.
      prefix: A prefix for files in the native (non-pkg) filegroups.
    """
    all_srcs = []
    no_driver_srcs = []
    pkg_srcs = []

    for dir, entries in install_dirs.items():
        for entry in entries:
            path = "{0}/{1}".format(dir, entry["name"])

            prefixed_path = "{0}/{1}".format(prefix, path)
            all_srcs.append(prefixed_path)
            if not entry["is_driver"]:
                no_driver_srcs.append(prefixed_path)

            pkg_path = path + ".pkg"
            pkg_srcs.append(pkg_path)

            if "target" in entry:
                if entry["executable"]:
                    symlink_file(
                        name = prefixed_path,
                        symlink_binary = entry["target"],
                    )
                    mode = "0755"
                else:
                    symlink_file(
                        name = prefixed_path,
                        symlink_label = entry["target"],
                    )
                    mode = "0644"
                pkg_files(
                    name = pkg_path,
                    srcs = [entry["target"]],
                    attributes = pkg_attributes(mode = mode),
                    renames = {entry["target"]: path},
                )
            elif "filegroup" in entry:
                symlink_filegroup(
                    name = prefixed_path,
                    out_prefix = prefixed_path,
                    srcs = [entry["filegroup"]],
                    remove_prefix = entry["remove_prefix"],
                )
                pkg_files(
                    name = pkg_path,
                    srcs = [prefixed_path],
                    strip_prefix = strip_prefix.from_pkg(prefix),
                )
            elif "symlink" in entry:
                symlink_to = "{0}/{1}/{2}".format(prefix, dir, entry["symlink"])

                # For bazel, we need to resolve relative symlinks.
                if "../" in symlink_to:
                    parts = symlink_to.split("/")
                    result = []
                    for part in parts:
                        if part == "..":
                            result = result[:-1]
                        else:
                            result.append(part)
                    symlink_to = "/".join(result)
                symlink_file(
                    name = prefixed_path,
                    symlink_binary = symlink_to,
                )

                # For the distributed package, we retain relative symlinks.
                pkg_mklink(
                    name = pkg_path,
                    link_name = path,
                    target = entry["symlink"],
                )
            else:
                fail("Unrecognized structure: {0}".format(entry))
    native.filegroup(name = name, srcs = all_srcs)
    native.filegroup(name = no_driver_name, srcs = no_driver_srcs)
    pkg_filegroup(name = pkg_name, srcs = pkg_srcs)
