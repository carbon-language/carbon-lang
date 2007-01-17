
#
# RECURSE test
#
FORWARD success;
FORWARD failure;
: returner 17 RETURN ;
: MAIN returner 17 = IF success ELSE failure ENDIF ;
