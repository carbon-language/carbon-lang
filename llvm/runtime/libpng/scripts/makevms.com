$! make libpng under VMS
$!
$!
$! Check for MMK/MMS
$!
$! This procedure accepts one parameter (contrib), which causes it to build
$! the programs from the contrib directory instead of libpng.
$!
$ p1 = f$edit(p1,"UPCASE")
$ if p1 .eqs. "CONTRIB"
$ then
$   set def [.contrib.gregbook]
$   @makevms
$   set def [-.pngminus]
$   @makevms
$   set def [--]
$   exit
$ endif
$ Make = ""
$ If F$Search ("Sys$System:MMS.EXE") .nes. "" Then Make = "MMS"
$ If F$Type (MMK) .eqs. "STRING" Then Make = "MMK"
$!
$! Look for the compiler used
$!
$ zlibsrc = "[-.zlib]"
$ ccopt="/include=''zlibsrc'"
$ if f$getsyi("HW_MODEL").ge.1024
$ then
$  ccopt = "/prefix=all"+ccopt
$  comp  = "__decc__=1"
$  if f$trnlnm("SYS").eqs."" then define sys sys$library:
$ else
$  if f$search("SYS$SYSTEM:DECC$COMPILER.EXE").eqs.""
$   then
$    if f$trnlnm("SYS").eqs."" then define sys sys$library:
$    if f$search("SYS$SYSTEM:VAXC.EXE").eqs.""
$     then
$      comp  = "__gcc__=1"
$      CC :== GCC
$     else
$      comp = "__vaxc__=1"
$     endif
$   else
$    if f$trnlnm("SYS").eqs."" then define sys decc$library_include:
$    ccopt = "/decc/prefix=all"+ccopt
$    comp  = "__decc__=1"
$  endif
$ endif
$!
$! Build the thing plain or with mms/mmk
$!
$ write sys$output "Compiling Libpng sources ..."
$ if make.eqs.""
$  then
$   dele pngtest.obj;*
$   CALL MAKE png.OBJ "cc ''CCOPT' png" -
	png.c png.h pngconf.h
$   CALL MAKE pngpread.OBJ "cc ''CCOPT' pngpread" -
					 pngpread.c png.h pngconf.h
$   CALL MAKE pngset.OBJ "cc ''CCOPT' pngset" -
	pngset.c png.h pngconf.h
$   CALL MAKE pngget.OBJ "cc ''CCOPT' pngget" -
	pngget.c png.h pngconf.h
$   CALL MAKE pngread.OBJ "cc ''CCOPT' pngread" -
	pngread.c png.h pngconf.h
$   CALL MAKE pngpread.OBJ "cc ''CCOPT' pngpread" -
					 pngpread.c png.h pngconf.h
$   CALL MAKE pngrtran.OBJ "cc ''CCOPT' pngrtran" -
	pngrtran.c png.h pngconf.h
$   CALL MAKE pngrutil.OBJ "cc ''CCOPT' pngrutil" -
	pngrutil.c png.h pngconf.h
$   CALL MAKE pngerror.OBJ "cc ''CCOPT' pngerror" -
	pngerror.c png.h pngconf.h
$   CALL MAKE pngmem.OBJ "cc ''CCOPT' pngmem" -
	pngmem.c png.h pngconf.h
$   CALL MAKE pngrio.OBJ "cc ''CCOPT' pngrio" -
	pngrio.c png.h pngconf.h
$   CALL MAKE pngwio.OBJ "cc ''CCOPT' pngwio" -
	pngwio.c png.h pngconf.h
$   CALL MAKE pngtrans.OBJ "cc ''CCOPT' pngtrans" -
	pngtrans.c png.h pngconf.h
$   CALL MAKE pngwrite.OBJ "cc ''CCOPT' pngwrite" -
	pngwrite.c png.h pngconf.h
$   CALL MAKE pngwtran.OBJ "cc ''CCOPT' pngwtran" -
	pngwtran.c png.h pngconf.h
$   CALL MAKE pngwutil.OBJ "cc ''CCOPT' pngwutil" -
	pngwutil.c png.h pngconf.h
$   write sys$output "Building Libpng ..."
$   CALL MAKE libpng.OLB "lib/crea libpng.olb *.obj" *.OBJ
$   write sys$output "Building pngtest..."
$   CALL MAKE pngtest.OBJ "cc ''CCOPT' pngtest" -
	pngtest.c png.h pngconf.h
$   call make pngtest.exe -
	"LINK pngtest,libpng.olb/lib,''zlibsrc'libz.olb/lib" -
	pngtest.obj libpng.olb
$   write sys$output "Testing Libpng..."
$   run pngtest
$  else
$   if f$search("DESCRIP.MMS") .eqs. "" then copy/nolog [.SCRIPTS]DESCRIP.MMS []
$   'make'/macro=('comp',zlibsrc='zlibsrc')
$  endif
$ write sys$output "Libpng build completed"
$ exit
$!
$!
$MAKE: SUBROUTINE   !SUBROUTINE TO CHECK DEPENDENCIES
$ V = 'F$Verify(0)
$! P1 = What we are trying to make
$! P2 = Command to make it
$! P3 - P8  What it depends on
$
$ If F$Search(P1) .Eqs. "" Then Goto Makeit
$ Time = F$CvTime(F$File(P1,"RDT"))
$arg=3
$Loop:
$       Argument = P'arg
$       If Argument .Eqs. "" Then Goto Exit
$       El=0
$Loop2:
$       File = F$Element(El," ",Argument)
$       If File .Eqs. " " Then Goto Endl
$       AFile = ""
$Loop3:
$       OFile = AFile
$       AFile = F$Search(File)
$       If AFile .Eqs. "" .Or. AFile .Eqs. OFile Then Goto NextEl
$       If F$CvTime(F$File(AFile,"RDT")) .Ges. Time Then Goto Makeit
$       Goto Loop3
$NextEL:
$       El = El + 1
$       Goto Loop2
$EndL:
$ arg=arg+1
$ If arg .Le. 8 Then Goto Loop
$ Goto Exit
$
$Makeit:
$ VV=F$VERIFY(0)
$ write sys$output P2
$ 'P2
$ VV='F$Verify(VV)
$Exit:
$ If V Then Set Verify
$ENDSUBROUTINE
