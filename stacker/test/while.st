#
# WHILE test
#
FORWARD success;
FORWARD failure;
: body "." >s DROP -- ; 
: do_while WHILE body END  ;
: MAIN 20 do_while 0 = IF success ELSE failure ENDIF;
