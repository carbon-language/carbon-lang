llvm-dsymutil - manipulate archived DWARF debug symbol files
============================================================

SYNOPSIS
--------

:program:`llvm-dsymutil` [*options*] [*filename*]

DESCRIPTION
-----------

:program:`llvm-dsymutil` links the DWARF debug information found in the object
files for the executable input file by using debug symbols information
contained in its symbol table.

OPTIONS
-------
.. option:: -arch=<string>

            Link DWARF debug information only for specified CPU architecture
            types. This option can be specified multiple times, once for each
            desired architecture.  All cpu architectures will be linked by
            default.

.. option:: -dump-debug-map

            Parse and dump the debug map to standard output. Not DWARF link
            will take place.

.. option:: -f, -flat

            Produce a flat dSYM file (not a bundle).

.. option:: -no-odr

            Do not use ODR (One Definition Rule) for type uniquing.

.. option:: -no-output

            Do the link in memory, but do not emit the result file.

.. option:: -no-swiftmodule-timestamp

            Don't check timestamp for swiftmodule files.

.. option:: -j <n>, -num-threads=<n>

            Specifies the maximum number (n) of simultaneous threads to use
            when linking multiple architectures.

.. option:: -o=<filename>

            Specify the output file. default: <input file>.dwarf

.. option:: -oso-prepend-path=<path>

            Specify a directory to prepend to the paths of object files.

.. option:: -s, -symtab

            Dumps the symbol table found in executable or object file(s) and
            exits.

.. option:: -v, -verbose

            Verbosity level

.. option:: --version

            Display the version of the tool.

.. option:: -y

            Treat the input file is a YAML debug map rather than a binary.


EXIT STATUS
-----------

:program:`llvm-dsymutil` returns 0 if the DWARF debug information was linked
successfully. Otherwise, it returns 1.

SEE ALSO
--------

:manpage:`llvm-dwarfdump(1)`
