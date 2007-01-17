#
# MIN test
#
FORWARD success;
FORWARD failure;
: step2 1 2 MIN 1 = IF success ELSE failure ENDIF ;
: MAIN 2 1 MIN 1 = IF step2 ELSE failure ENDIF ;
