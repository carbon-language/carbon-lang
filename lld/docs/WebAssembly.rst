WebAssembly lld port
====================

The WebAssembly version of lld takes WebAssembly binaries as inputs and produces
a WebAssembly binary as its output.  For the most part it tries to mimic the
behaviour of traditional ELF linkers and specifically the ELF lld port.  Where
possible that command line flags and the semantics should be the same.


Object file format
------------------

The format the input object files that lld expects is specified as part of the
the WebAssembly tool conventions
https://github.com/WebAssembly/tool-conventions/blob/master/Linking.md.

This is object format that the llvm will produce when run with the
``wasm32-unknown-unknown`` target.  To build llvm with WebAssembly support
currently requires enabling the experimental backed using
``-DLLVM_EXPERIMENTAL_TARGETS_TO_BUILD=WebAssembly``.


Usage
-----

The WebAssembly version of lld is installed as **wasm-ld**.  It shared many 
common linker flags with **ld.lld** but also includes several
WebAssembly-specific options:

.. option:: --no-entry

  Don't search for the entry point symbol (by default ``_start``).

.. option:: --export-table

  Export the function table to the environment.

.. option:: --import-table

  Import the function table from the environment.

.. option:: --export-all

  Export all symbols (normally combined with --no-gc-sections)

.. option:: --[no-]export-default

  Export symbols marked as 'default' visibility.  Default: true

.. option:: --global-base=<value>

  Address at which to place global data

.. option:: --[no-]merge-data-segments

  Enable/Disble merging of data segments.  Default: true

.. option:: --stack-first

  Place stack at start of linear memory rather than after data

.. option:: --compress-relocations

  Compress the relocation targets in the code section.

.. option:: --allow-undefined

  Allow undefined symbols in linked binary

.. option:: --import-memory

  Import memory from the environment

.. option:: --initial-memory=<value>

  Initial size of the linear memory. Default: static data size

.. option:: --max-memory=<value>

  Maximum size of the linear memory. Default: unlimited

By default the function table is neither imported nor exported.

Symbols are exported if they are marked as ``visibility=default`` at compile
time or if they are included on the command line via ``--export``.

Since WebAssembly is designed with size in mind the linker defaults to
``--gc-sections`` which means that all un-used functions and data segments will
be stripped from the binary.

The symbols which are preserved by default are:

- The entry point (by default ``_start``).
- Any symbol which is to be exported.
- Any symbol transitively referenced by the above.


Missing features
----------------

- Merging of data section similiar to ``SHF_MERGE`` in the ELF world is not
  supported.
- No support for creating shared libaries.  The spec for shared libraries in
  WebAssembly is still in flux:
  https://github.com/WebAssembly/tool-conventions/blob/master/DynamicLinking.md
