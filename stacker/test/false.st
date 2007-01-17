#
# FALSE test
#
FORWARD success;
FORWARD failure;
: MAIN FALSE 0 = IF success ELSE failure ENDIF ;
