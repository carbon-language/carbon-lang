ELF ARM
~~~~~~~~~~~

Unimplemented Features
######################

* Static executable linking - in progress
* Dynamic executable linking
* DSO linking
* PLT entries' generation for images larger than 2^28 bytes (see Sec. A.3 of the ELF reference)
* ARM and Thumb interworking (see http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.dui0203j/Bcghfebi.html)
* .ARM.exidx section handling
* -init/-fini options
* Lots of relocations

Unimplemented Relocations
#########################

All of these relocations are defined in:
http://infocenter.arm.com/help/topic/com.arm.doc.ihi0044e/IHI0044E_aaelf.pdf
