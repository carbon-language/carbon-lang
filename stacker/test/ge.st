#
# GE test
#
FORWARD success;
FORWARD failure;
: phase2 49 49 >= IF success ELSE failure ENDIF ;
: MAIN 7 49 >= IF phase2 ELSE failure ENDIF ;
