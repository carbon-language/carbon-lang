ELF x86-64
~~~~~~~~~~

Unimplemented Features
######################

* Code models other than the small code model
* Fully dynamic TLS
* TLS strength reduction

Unimplemented Relocations
#########################

All of these relocations are defined in:
http://www.x86-64.org/documentation/abi.pdf

Trivial Relocs
<<<<<<<<<<<<<<

These are very simple relocation calculations to implement.
See lib/ReaderWriter/ELF/X86_64/X86_64RelocationHandler.cpp

* R_X86_64_16
* R_X86_64_PC16
* R_X86_64_8
* R_X86_64_PC8
* R_X86_64_PC64
* R_X86_64_SIZE32
* R_X86_64_SIZE64
* R_X86_64_GOTPC32 (this relocation requires there to be a __GLOBAL_OFFSET_TABLE__)

Global Offset Table Relocs
<<<<<<<<<<<<<<<<<<<<<<<<<<

* R_X86_64_GOTOFF32
* R_X86_64_GOTOFF64

Global Dynamic Thread Local Storage Relocs
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

These relocations will take more effort to implement. Their implementation
should be contained in lib/ReaderWriter/ELF/X86_64/{X86_64RelocationPass.cpp,X86_64RelocationHandler.cpp}.

Documentation on these relocations can be found in:
http://www.akkadia.org/drepper/tls.pdf

* R_X86_64_DTPMOD64
* R_X86_64_DTPOFF64
* R_X86_64_TLSGD
* R_X86_64_GOTPC32_TLSDESC
* R_X86_64_TLSDESC_CALL
* R_X86_64_TLSDESC
