
To run this utility:

For a list of options:
        llc -h

To see the generated machine instructions:
        llc -debug_select 1 <bytecode-file>

I left that as a "debugging" option since it is not real code.
Use 2 instead of 1 to dump the mapping between LLVM instructions and
Machine instructions, and 5 to see the patterns chosen by BURG.
These outputs aren't very clean.

