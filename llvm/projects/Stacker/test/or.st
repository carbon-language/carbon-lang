#
# OR test
#
FORWARD success;
FORWARD failure;
: phase2 0 0 OR 0 = IF success ELSE failure ENDIF ;
: MAIN 7 8 OR 15 = IF phase2 ELSE failure ENDIF ;
