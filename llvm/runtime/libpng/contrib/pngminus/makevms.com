$!------------------------------------------------------------------------------
$! make Contrib programs of libpng under OpenVMS
$!
$!
$! Look for the compiler used
$!
$ zlibsrc = "[---.zlib]"
$ ccopt="/include=(''zlibsrc',[--])"
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
$ open/write lopt lib.opt
$ write lopt "[--]libpng.olb/lib"
$ write lopt "''zlibsrc'libz.olb/lib"
$ close lopt
$ open/write xopt x11.opt
$ write xopt "sys$library:decw$xlibshr.exe/share"
$ close xopt
$ write sys$output "Compiling PNG contrib programs ..."
$   write sys$output "Building pnm2png..."
$   CALL MAKE pnm2png.OBJ "cc ''CCOPT' pnm2png" -
	pnm2png.c
$   call make pnm2png.exe -
	"LINK pnm2png,lib.opt/opt" -
	pnm2png.obj
$   write sys$output "Building png2pnm..."
$   CALL MAKE png2pnm.OBJ "cc ''CCOPT' png2pnm" -
	png2pnm.c
$   call make png2pnm.exe -
	"LINK png2pnm,lib.opt/opt" -
	png2pnm.obj
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
