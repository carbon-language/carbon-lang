======
Driver
======

.. contents::
   :local:

Introduction
============

This document describes the lld driver. The purpose of this document is to
describe both the motivation and design goals for the driver, as well as details
of the internal implementation.

Overview
========

The lld driver is designed to support a number of different command line
interfaces. The main interfaces we plan to support are binutils ld, Apple's
ld64, and Microsoft's link.exe.

Flavors
-------

Each of these different interfaces is refered to as a flavor. There is also an
extra flavor for ``lld -core``. This is eqivilent to ``-cc1`` in clang.
The current flavors are.

* ld
* ld64
* link
* core

Selecting a Flavor
^^^^^^^^^^^^^^^^^^

There are two different ways to tell lld which flavor to be. They are checked in
order, so the second overrides the first. The first is to symlink :program:`lld`
as :program:`lld-{flavor}` or just :program:`{flavor}`. You can also specify
it as the first command line argument using ``-flavor``::

  $ lld -flavor ld

There is a shortcut for ``-flavor core`` as ``-core``.

Argument translation and lld -core
----------------------------------

Due to the different driver flavors and the need to write portable tests, there
is ``lld -core``. Driver flavors translate options into ``lld -core`` options.
These options are then forwarded to ``lld -core`` to run the actual link. The
options passed to ``lld -core`` can be seen by passing ``-###`` to any driver
flavor.

Targets
-------

The ``-target <llvm-triple>`` option can be passed to any driver flavor to
link for a specific target. You can also prefix the :program:`lld` symlink with
a target triple to default to that target. If neither of these is set, the
default target is the target LLVM was configured for.

Adding an Option
================

#. Add the option to the desired :file:`lib/Driver/{flavor}Options.td`.

#. If there is no ``lld -core`` option, add the option to
   :file:`lib/Driver/CoreOptions.td`. All ``lld -core`` start with a single -
   and if they have a value, it is joined with a =. ``lld -core`` options should
   have sensible, non-abbrivated names and should be shared between flavors
   where possible.

#. Modify the ``{flavor}Driver::transform`` function to transform the added
   option into one or more core options.

#. Add the option to :cpp:class:`lld::LinkerOptions` in
   :file:`include/lld/Driver/LinkerOptions.h` and modify the move constructor to
   move the option value.

#. Modify :cpp:func:`lld::parseCoreArgs` in :file:`lib/Driver/Drivers.cpp` to
   fill the :cpp:class:`lld::LinkerOptions` with the new option.

#. Modify lld to use the new option.

Adding a Flavor
===============

#. Add an entry for the flavor in :file:`include/lld/Driver/Driver.h` to
   :cpp:class:`lld::Driver::Flavor`.

#. Add an entry in :file:`tools/lld/lld.cpp` to
   :cpp:func:`lld::Driver::strToFlavor`. This allows the
   flavor to be selected via symlink and :option:`-flavor`.

#. Add a tablegen file called :file:`lib/Driver/{flavor}Options.td` that
   describes the options. If the options are a superset of another driver, that
   driver's td file can simply be included. The :file:`{flavor}Options.td` file
   must also be added to :file:`lib/Driver/CMakeListst.txt`.

#. Add a ``{flavor}::{flavor}OptTable`` as a subclass of
   :cpp:class:`llvm::opt::OptTable` in :file:`lib/Driver/Drivers.cpp`.

#. Add a ``{flavor}Driver`` as a subclass of :cpp:class:`lld::Driver`
   in :file:`lib/Driver/Drivers.cpp`. It must have a :cpp:func:`transform`
   function which takes argc/argv and returns a ``lld -core`` ArgList.

#. Modify :cpp:func:`Driver::create` to create an instance of the new driver.
