llvm-nm - list LLVM bitcode file's symbol table
===============================================


SYNOPSIS
--------


**llvm-nm** [*options*] [*filenames...*]


DESCRIPTION
-----------


The **llvm-nm** utility lists the names of symbols from the LLVM bitcode files,
or **ar** archives containing LLVM bitcode files, named on the command line.
Each symbol is listed along with some simple information about its provenance.
If no file name is specified, or *-* is used as a file name, **llvm-nm** will
process a bitcode file on its standard input stream.

**llvm-nm**'s default output format is the traditional BSD **nm** output format.
Each such output record consists of an (optional) 8-digit hexadecimal address,
followed by a type code character, followed by a name, for each symbol. One
record is printed per line; fields are separated by spaces. When the address is
omitted, it is replaced by 8 spaces.

Type code characters currently supported, and their meanings, are as follows:


U

 Named object is referenced but undefined in this bitcode file



C

 Common (multiple definitions link together into one def)



W

 Weak reference (multiple definitions link together into zero or one definitions)



t

 Local function (text) object



T

 Global function (text) object



d

 Local data object



D

 Global data object



?

 Something unrecognizable



Because LLVM bitcode files typically contain objects that are not considered to
have addresses until they are linked into an executable image or dynamically
compiled "just-in-time", **llvm-nm** does not print an address for any symbol,
even symbols which are defined in the bitcode file.


OPTIONS
-------



**-P**

 Use POSIX.2 output format. Alias for **--format=posix**.



**-B**    (default)

 Use BSD output format. Alias for **--format=bsd**.



**-help**

 Print a summary of command-line options and their meanings.



**--defined-only**

 Print only symbols defined in this bitcode file (as opposed to
 symbols which may be referenced by objects in this file, but not
 defined in this file.)



**--extern-only**, **-g**

 Print only symbols whose definitions are external; that is, accessible
 from other bitcode files.



**--undefined-only**, **-u**

 Print only symbols referenced but not defined in this bitcode file.




 Select an output format; *fmt* may be *sysv*, *posix*, or *bsd*. The
 default is *bsd*.




BUGS
----


**llvm-nm** cannot demangle C++ mangled names, like GNU **nm** can.


EXIT STATUS
-----------


**llvm-nm** exits with an exit code of zero.


SEE ALSO
--------


llvm-dis|llvm-dis, ar(1), nm(1)
