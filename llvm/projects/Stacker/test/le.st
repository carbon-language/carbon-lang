#
# LE test
#
FORWARD success;
FORWARD failure;
: phase2 49 49 <= IF success ELSE failure ENDIF ;
: MAIN 49 7 <= IF phase2 ELSE failure ENDIF ;
