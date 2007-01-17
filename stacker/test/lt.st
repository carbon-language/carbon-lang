#
# LT test
#
FORWARD success;
FORWARD failure;
: phase2 49 7 < IF success ELSE failure ENDIF ;
: MAIN 7 49 < IF failure ELSE phase2 ENDIF ;
