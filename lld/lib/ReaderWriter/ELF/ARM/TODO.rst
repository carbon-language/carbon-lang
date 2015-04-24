ELF ARM
~~~~~~~~~~~

Unimplemented Features
######################

* DSO linking
* C++ code linking
* PLT entries' generation for images larger than 2^28 bytes (see Sec. A.3 of the ARM ELF reference)
* ARM/Thumb interwork veneers in position-independent code
* .ARM.exidx section (exception handling)
* -init/-fini options
* Proper debug information (DWARF data)
* TLS relocations for dynamic models
* Lots of other relocations

Unimplemented Relocations
#########################

All of these relocations are defined in:
http://infocenter.arm.com/help/topic/com.arm.doc.ihi0044e/IHI0044E_aaelf.pdf
