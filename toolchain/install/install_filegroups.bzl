# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rules for constructing install information."""

load("@rules_pkg//pkg:mappings.bzl", "pkg_attributes", "pkg_filegroup", "pkg_files", "pkg_mklink", "strip_prefix")
load("symlink_helpers.bzl", "symlink_file", "symlink_filegroup")

def install_filegroup(name, filegroup_target):
    """Adds a filegroup for install.

    Used in the `install_dirs` dict.

    Args:
      name: The base directory for the filegroup.
      filegroup_target: A relative path for the symlink.
    """
    return {
        "filegroup": filegroup_target,
        "for_driver": False,
        "name": name,
    }

def install_symlink(name, symlink_to):
    """Adds a symlink for install.

    Used in the `install_dirs` dict.

    Args:
      name: The filename to use.
      symlink_to: A relative path for the symlink.
    """
    return {
        "for_driver": False,
        "name": name,
        "symlink": symlink_to,
    }

def install_target(name, target, executable = False, for_driver = False):
    """Adds a target for install.

    Used in the `install_dirs` dict.

    Args:
      name: The filename to use.
      target: The bazel target being installed.
      executable: True if executable.
      for_driver: False if it should be included in the `no_driver_name`
        filegroup.
    """
    return {
        "executable": executable,
        "for_driver": for_driver,
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
            if not entry["for_driver"]:
                no_driver_srcs.append(prefixed_path)

            pkg_path = path + ".pkg"
            pkg_srcs.append(pkg_path)

            if "target" in entry:
                symlink_file(
                    name = prefixed_path,
                    symlink_label = entry["target"],
                )
                if entry["executable"]:
                    mode = "0755"
                else:
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
                )
                pkg_files(
                    name = pkg_path,
                    srcs = [prefixed_path],
                    strip_prefix = strip_prefix.from_pkg(prefix),
                )
            elif "symlink" in entry:
                symlink_file(
                    name = prefixed_path,
                    symlink_relative = entry["symlink"],
                )
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
