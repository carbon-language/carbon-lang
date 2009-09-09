; RUN: llc < %s
; PR4534

; ModuleID = 'tango.net.ftp.FtpClient.bc'
target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f128:64:128"
target triple = "powerpc-apple-darwin9.6.0"
	%"byte[]" = type { i32, i8* }
@.str167 = external constant [11 x i8]		; <[11 x i8]*> [#uses=1]
@.str170 = external constant [11 x i8]		; <[11 x i8]*> [#uses=2]
@.str171 = external constant [5 x i8]		; <[5 x i8]*> [#uses=1]
@llvm.used = appending global [1 x i8*] [i8* bitcast (void (%"byte[]")* @foo to i8*)], section "llvm.metadata"		; <[1 x i8*]*> [#uses=0]

define fastcc void @foo(%"byte[]" %line_arg) {
entry:
	%line_arg830 = extractvalue %"byte[]" %line_arg, 0		; <i32> [#uses=12]
	%line_arg831 = extractvalue %"byte[]" %line_arg, 1		; <i8*> [#uses=17]
	%t5 = load i8* %line_arg831		; <i8> [#uses=1]
	br label %forcondi

forcondi:		; preds = %forbodyi, %entry
	%l.0i = phi i32 [ 10, %entry ], [ %t4i, %forbodyi ]		; <i32> [#uses=2]
	%p.0i = phi i8* [ getelementptr ([11 x i8]* @.str167, i32 0, i32 -1), %entry ], [ %t7i, %forbodyi ]		; <i8*> [#uses=1]
	%t4i = add i32 %l.0i, -1		; <i32> [#uses=1]
	%t5i = icmp eq i32 %l.0i, 0		; <i1> [#uses=1]
	br i1 %t5i, label %forcond.i, label %forbodyi

forbodyi:		; preds = %forcondi
	%t7i = getelementptr i8* %p.0i, i32 1		; <i8*> [#uses=2]
	%t8i = load i8* %t7i		; <i8> [#uses=1]
	%t12i = icmp eq i8 %t8i, %t5		; <i1> [#uses=1]
	br i1 %t12i, label %forcond.i, label %forcondi

forcond.i:		; preds = %forbody.i, %forbodyi, %forcondi
	%storemerge.i = phi i32 [ %t106.i, %forbody.i ], [ 1, %forcondi ], [ 1, %forbodyi ]		; <i32> [#uses=1]
	%t77.i286 = phi i1 [ %phit3, %forbody.i ], [ false, %forcondi ], [ false, %forbodyi ]		; <i1> [#uses=1]
	br i1 %t77.i286, label %forcond.i295, label %forbody.i

forbody.i:		; preds = %forcond.i
	%t106.i = add i32 %storemerge.i, 1		; <i32> [#uses=2]
	%phit3 = icmp ugt i32 %t106.i, 3		; <i1> [#uses=1]
	br label %forcond.i

forcond.i295:		; preds = %forbody.i301, %forcond.i
	%storemerge.i292 = phi i32 [ %t106.i325, %forbody.i301 ], [ 4, %forcond.i ]		; <i32> [#uses=1]
	%t77.i293 = phi i1 [ %phit2, %forbody.i301 ], [ false, %forcond.i ]		; <i1> [#uses=1]
	br i1 %t77.i293, label %forcond.i332, label %forbody.i301

forbody.i301:		; preds = %forcond.i295
	%t106.i325 = add i32 %storemerge.i292, 1		; <i32> [#uses=2]
	%phit2 = icmp ugt i32 %t106.i325, 6		; <i1> [#uses=1]
	br label %forcond.i295

forcond.i332:		; preds = %forbody.i338, %forcond.i295
	%storemerge.i329 = phi i32 [ %t106.i362, %forbody.i338 ], [ 7, %forcond.i295 ]		; <i32> [#uses=3]
	%t77.i330 = phi i1 [ %phit1, %forbody.i338 ], [ false, %forcond.i295 ]		; <i1> [#uses=1]
	br i1 %t77.i330, label %wcond.i370, label %forbody.i338

forbody.i338:		; preds = %forcond.i332
	%t106.i362 = add i32 %storemerge.i329, 1		; <i32> [#uses=2]
	%phit1 = icmp ugt i32 %t106.i362, 9		; <i1> [#uses=1]
	br label %forcond.i332

wcond.i370:		; preds = %wbody.i372, %forcond.i332
	%.frame.0.11 = phi i32 [ %t18.i371.c, %wbody.i372 ], [ %storemerge.i329, %forcond.i332 ]		; <i32> [#uses=2]
	%t3.i368 = phi i32 [ %t18.i371.c, %wbody.i372 ], [ %storemerge.i329, %forcond.i332 ]		; <i32> [#uses=5]
	%t4.i369 = icmp ult i32 %t3.i368, %line_arg830		; <i1> [#uses=1]
	br i1 %t4.i369, label %andand.i378, label %wcond22.i383

wbody.i372:		; preds = %andand.i378
	%t18.i371.c = add i32 %t3.i368, 1		; <i32> [#uses=2]
	br label %wcond.i370

andand.i378:		; preds = %wcond.i370
	%t11.i375 = getelementptr i8* %line_arg831, i32 %t3.i368		; <i8*> [#uses=1]
	%t12.i376 = load i8* %t11.i375		; <i8> [#uses=1]
	%t14.i377 = icmp eq i8 %t12.i376, 32		; <i1> [#uses=1]
	br i1 %t14.i377, label %wbody.i372, label %wcond22.i383

wcond22.i383:		; preds = %wbody23.i385, %andand.i378, %wcond.i370
	%.frame.0.10 = phi i32 [ %t50.i384, %wbody23.i385 ], [ %.frame.0.11, %wcond.i370 ], [ %.frame.0.11, %andand.i378 ]		; <i32> [#uses=2]
	%t49.i381 = phi i32 [ %t50.i384, %wbody23.i385 ], [ %t3.i368, %wcond.i370 ], [ %t3.i368, %andand.i378 ]		; <i32> [#uses=5]
	%t32.i382 = icmp ult i32 %t49.i381, %line_arg830		; <i1> [#uses=1]
	br i1 %t32.i382, label %andand33.i391, label %wcond54.i396

wbody23.i385:		; preds = %andand33.i391
	%t50.i384 = add i32 %t49.i381, 1		; <i32> [#uses=2]
	br label %wcond22.i383

andand33.i391:		; preds = %wcond22.i383
	%t42.i388 = getelementptr i8* %line_arg831, i32 %t49.i381		; <i8*> [#uses=1]
	%t43.i389 = load i8* %t42.i388		; <i8> [#uses=1]
	%t45.i390 = icmp eq i8 %t43.i389, 32		; <i1> [#uses=1]
	br i1 %t45.i390, label %wcond54.i396, label %wbody23.i385

wcond54.i396:		; preds = %wbody55.i401, %andand33.i391, %wcond22.i383
	%.frame.0.9 = phi i32 [ %t82.i400, %wbody55.i401 ], [ %.frame.0.10, %wcond22.i383 ], [ %.frame.0.10, %andand33.i391 ]		; <i32> [#uses=2]
	%t81.i394 = phi i32 [ %t82.i400, %wbody55.i401 ], [ %t49.i381, %wcond22.i383 ], [ %t49.i381, %andand33.i391 ]		; <i32> [#uses=3]
	%t64.i395 = icmp ult i32 %t81.i394, %line_arg830		; <i1> [#uses=1]
	br i1 %t64.i395, label %andand65.i407, label %wcond.i716

wbody55.i401:		; preds = %andand65.i407
	%t82.i400 = add i32 %t81.i394, 1		; <i32> [#uses=2]
	br label %wcond54.i396

andand65.i407:		; preds = %wcond54.i396
	%t74.i404 = getelementptr i8* %line_arg831, i32 %t81.i394		; <i8*> [#uses=1]
	%t75.i405 = load i8* %t74.i404		; <i8> [#uses=1]
	%t77.i406 = icmp eq i8 %t75.i405, 32		; <i1> [#uses=1]
	br i1 %t77.i406, label %wbody55.i401, label %wcond.i716

wcond.i716:		; preds = %wbody.i717, %andand65.i407, %wcond54.i396
	%.frame.0.0 = phi i32 [ %t18.i.c829, %wbody.i717 ], [ %.frame.0.9, %wcond54.i396 ], [ %.frame.0.9, %andand65.i407 ]		; <i32> [#uses=7]
	%t4.i715 = icmp ult i32 %.frame.0.0, %line_arg830		; <i1> [#uses=1]
	br i1 %t4.i715, label %andand.i721, label %wcond22.i724

wbody.i717:		; preds = %andand.i721
	%t18.i.c829 = add i32 %.frame.0.0, 1		; <i32> [#uses=1]
	br label %wcond.i716

andand.i721:		; preds = %wcond.i716
	%t11.i718 = getelementptr i8* %line_arg831, i32 %.frame.0.0		; <i8*> [#uses=1]
	%t12.i719 = load i8* %t11.i718		; <i8> [#uses=1]
	%t14.i720 = icmp eq i8 %t12.i719, 32		; <i1> [#uses=1]
	br i1 %t14.i720, label %wbody.i717, label %wcond22.i724

wcond22.i724:		; preds = %wbody23.i726, %andand.i721, %wcond.i716
	%.frame.0.1 = phi i32 [ %t50.i725, %wbody23.i726 ], [ %.frame.0.0, %wcond.i716 ], [ %.frame.0.0, %andand.i721 ]		; <i32> [#uses=2]
	%t49.i722 = phi i32 [ %t50.i725, %wbody23.i726 ], [ %.frame.0.0, %wcond.i716 ], [ %.frame.0.0, %andand.i721 ]		; <i32> [#uses=5]
	%t32.i723 = icmp ult i32 %t49.i722, %line_arg830		; <i1> [#uses=1]
	br i1 %t32.i723, label %andand33.i731, label %wcond54.i734

wbody23.i726:		; preds = %andand33.i731
	%t50.i725 = add i32 %t49.i722, 1		; <i32> [#uses=2]
	br label %wcond22.i724

andand33.i731:		; preds = %wcond22.i724
	%t42.i728 = getelementptr i8* %line_arg831, i32 %t49.i722		; <i8*> [#uses=1]
	%t43.i729 = load i8* %t42.i728		; <i8> [#uses=1]
	%t45.i730 = icmp eq i8 %t43.i729, 32		; <i1> [#uses=1]
	br i1 %t45.i730, label %wcond54.i734, label %wbody23.i726

wcond54.i734:		; preds = %wbody55.i736, %andand33.i731, %wcond22.i724
	%.frame.0.2 = phi i32 [ %t82.i735, %wbody55.i736 ], [ %.frame.0.1, %wcond22.i724 ], [ %.frame.0.1, %andand33.i731 ]		; <i32> [#uses=2]
	%t81.i732 = phi i32 [ %t82.i735, %wbody55.i736 ], [ %t49.i722, %wcond22.i724 ], [ %t49.i722, %andand33.i731 ]		; <i32> [#uses=3]
	%t64.i733 = icmp ult i32 %t81.i732, %line_arg830		; <i1> [#uses=1]
	br i1 %t64.i733, label %andand65.i740, label %wcond.i750

wbody55.i736:		; preds = %andand65.i740
	%t82.i735 = add i32 %t81.i732, 1		; <i32> [#uses=2]
	br label %wcond54.i734

andand65.i740:		; preds = %wcond54.i734
	%t74.i737 = getelementptr i8* %line_arg831, i32 %t81.i732		; <i8*> [#uses=1]
	%t75.i738 = load i8* %t74.i737		; <i8> [#uses=1]
	%t77.i739 = icmp eq i8 %t75.i738, 32		; <i1> [#uses=1]
	br i1 %t77.i739, label %wbody55.i736, label %wcond.i750

wcond.i750:		; preds = %wbody.i752, %andand65.i740, %wcond54.i734
	%.frame.0.3 = phi i32 [ %t18.i751.c, %wbody.i752 ], [ %.frame.0.2, %wcond54.i734 ], [ %.frame.0.2, %andand65.i740 ]		; <i32> [#uses=11]
	%t4.i749 = icmp ult i32 %.frame.0.3, %line_arg830		; <i1> [#uses=1]
	br i1 %t4.i749, label %andand.i758, label %wcond22.i761

wbody.i752:		; preds = %andand.i758
	%t18.i751.c = add i32 %.frame.0.3, 1		; <i32> [#uses=1]
	br label %wcond.i750

andand.i758:		; preds = %wcond.i750
	%t11.i755 = getelementptr i8* %line_arg831, i32 %.frame.0.3		; <i8*> [#uses=1]
	%t12.i756 = load i8* %t11.i755		; <i8> [#uses=1]
	%t14.i757 = icmp eq i8 %t12.i756, 32		; <i1> [#uses=1]
	br i1 %t14.i757, label %wbody.i752, label %wcond22.i761

wcond22.i761:		; preds = %wbody23.i763, %andand.i758, %wcond.i750
	%.frame.0.4 = phi i32 [ %t50.i762, %wbody23.i763 ], [ %.frame.0.3, %wcond.i750 ], [ %.frame.0.3, %andand.i758 ]		; <i32> [#uses=2]
	%t49.i759 = phi i32 [ %t50.i762, %wbody23.i763 ], [ %.frame.0.3, %wcond.i750 ], [ %.frame.0.3, %andand.i758 ]		; <i32> [#uses=7]
	%t32.i760 = icmp ult i32 %t49.i759, %line_arg830		; <i1> [#uses=1]
	br i1 %t32.i760, label %andand33.i769, label %wcond54.i773

wbody23.i763:		; preds = %andand33.i769
	%t50.i762 = add i32 %t49.i759, 1		; <i32> [#uses=2]
	br label %wcond22.i761

andand33.i769:		; preds = %wcond22.i761
	%t42.i766 = getelementptr i8* %line_arg831, i32 %t49.i759		; <i8*> [#uses=1]
	%t43.i767 = load i8* %t42.i766		; <i8> [#uses=1]
	%t45.i768 = icmp eq i8 %t43.i767, 32		; <i1> [#uses=1]
	br i1 %t45.i768, label %wcond54.i773, label %wbody23.i763

wcond54.i773:		; preds = %wbody55.i775, %andand33.i769, %wcond22.i761
	%.frame.0.5 = phi i32 [ %t82.i774, %wbody55.i775 ], [ %.frame.0.4, %wcond22.i761 ], [ %.frame.0.4, %andand33.i769 ]		; <i32> [#uses=1]
	%t81.i770 = phi i32 [ %t82.i774, %wbody55.i775 ], [ %t49.i759, %wcond22.i761 ], [ %t49.i759, %andand33.i769 ]		; <i32> [#uses=3]
	%t64.i771 = icmp ult i32 %t81.i770, %line_arg830		; <i1> [#uses=1]
	br i1 %t64.i771, label %andand65.i780, label %Dt3net3ftp9FClient13FConnection13pListLineMFAaZS5t3net3ftp9FClient11FFileInfo10p_wordMFZAa.exit786

wbody55.i775:		; preds = %andand65.i780
	%t82.i774 = add i32 %t81.i770, 1		; <i32> [#uses=2]
	br label %wcond54.i773

andand65.i780:		; preds = %wcond54.i773
	%t74.i777 = getelementptr i8* %line_arg831, i32 %t81.i770		; <i8*> [#uses=1]
	%t75.i778 = load i8* %t74.i777		; <i8> [#uses=1]
	%t77.i779 = icmp eq i8 %t75.i778, 32		; <i1> [#uses=1]
	br i1 %t77.i779, label %wbody55.i775, label %Dt3net3ftp9FClient13FConnection13pListLineMFAaZS5t3net3ftp9FClient11FFileInfo10p_wordMFZAa.exit786

Dt3net3ftp9FClient13FConnection13pListLineMFAaZS5t3net3ftp9FClient11FFileInfo10p_wordMFZAa.exit786:		; preds = %andand65.i780, %wcond54.i773
	%t89.i782 = getelementptr i8* %line_arg831, i32 %.frame.0.3		; <i8*> [#uses=4]
	%t90.i783 = sub i32 %t49.i759, %.frame.0.3		; <i32> [#uses=2]
	br label %wcond.i792

wcond.i792:		; preds = %wbody.i794, %Dt3net3ftp9FClient13FConnection13pListLineMFAaZS5t3net3ftp9FClient11FFileInfo10p_wordMFZAa.exit786
	%.frame.0.6 = phi i32 [ %.frame.0.5, %Dt3net3ftp9FClient13FConnection13pListLineMFAaZS5t3net3ftp9FClient11FFileInfo10p_wordMFZAa.exit786 ], [ %t18.i793.c, %wbody.i794 ]		; <i32> [#uses=9]
	%t4.i791 = icmp ult i32 %.frame.0.6, %line_arg830		; <i1> [#uses=1]
	br i1 %t4.i791, label %andand.i800, label %wcond22.i803

wbody.i794:		; preds = %andand.i800
	%t18.i793.c = add i32 %.frame.0.6, 1		; <i32> [#uses=1]
	br label %wcond.i792

andand.i800:		; preds = %wcond.i792
	%t11.i797 = getelementptr i8* %line_arg831, i32 %.frame.0.6		; <i8*> [#uses=1]
	%t12.i798 = load i8* %t11.i797		; <i8> [#uses=1]
	%t14.i799 = icmp eq i8 %t12.i798, 32		; <i1> [#uses=1]
	br i1 %t14.i799, label %wbody.i794, label %wcond22.i803

wcond22.i803:		; preds = %wbody23.i805, %andand.i800, %wcond.i792
	%t49.i801 = phi i32 [ %t50.i804, %wbody23.i805 ], [ %.frame.0.6, %wcond.i792 ], [ %.frame.0.6, %andand.i800 ]		; <i32> [#uses=7]
	%t32.i802 = icmp ult i32 %t49.i801, %line_arg830		; <i1> [#uses=1]
	br i1 %t32.i802, label %andand33.i811, label %wcond54.i815

wbody23.i805:		; preds = %andand33.i811
	%t50.i804 = add i32 %t49.i801, 1		; <i32> [#uses=1]
	br label %wcond22.i803

andand33.i811:		; preds = %wcond22.i803
	%t42.i808 = getelementptr i8* %line_arg831, i32 %t49.i801		; <i8*> [#uses=1]
	%t43.i809 = load i8* %t42.i808		; <i8> [#uses=1]
	%t45.i810 = icmp eq i8 %t43.i809, 32		; <i1> [#uses=1]
	br i1 %t45.i810, label %wcond54.i815, label %wbody23.i805

wcond54.i815:		; preds = %wbody55.i817, %andand33.i811, %wcond22.i803
	%t81.i812 = phi i32 [ %t82.i816, %wbody55.i817 ], [ %t49.i801, %wcond22.i803 ], [ %t49.i801, %andand33.i811 ]		; <i32> [#uses=3]
	%t64.i813 = icmp ult i32 %t81.i812, %line_arg830		; <i1> [#uses=1]
	br i1 %t64.i813, label %andand65.i822, label %Dt3net3ftp9FClient13FConnection13pListLineMFAaZS5t3net3ftp9FClient11FFileInfo10p_wordMFZAa.exit828

wbody55.i817:		; preds = %andand65.i822
	%t82.i816 = add i32 %t81.i812, 1		; <i32> [#uses=1]
	br label %wcond54.i815

andand65.i822:		; preds = %wcond54.i815
	%t74.i819 = getelementptr i8* %line_arg831, i32 %t81.i812		; <i8*> [#uses=1]
	%t75.i820 = load i8* %t74.i819		; <i8> [#uses=1]
	%t77.i821 = icmp eq i8 %t75.i820, 32		; <i1> [#uses=1]
	br i1 %t77.i821, label %wbody55.i817, label %Dt3net3ftp9FClient13FConnection13pListLineMFAaZS5t3net3ftp9FClient11FFileInfo10p_wordMFZAa.exit828

Dt3net3ftp9FClient13FConnection13pListLineMFAaZS5t3net3ftp9FClient11FFileInfo10p_wordMFZAa.exit828:		; preds = %andand65.i822, %wcond54.i815
	%t89.i824 = getelementptr i8* %line_arg831, i32 %.frame.0.6		; <i8*> [#uses=4]
	%t90.i825 = sub i32 %t49.i801, %.frame.0.6		; <i32> [#uses=2]
	%t63 = load i8* %t89.i824		; <i8> [#uses=2]
	br label %forcondi622

forcondi622:		; preds = %forbodyi626, %Dt3net3ftp9FClient13FConnection13pListLineMFAaZS5t3net3ftp9FClient11FFileInfo10p_wordMFZAa.exit828
	%l.0i618 = phi i32 [ 10, %Dt3net3ftp9FClient13FConnection13pListLineMFAaZS5t3net3ftp9FClient11FFileInfo10p_wordMFZAa.exit828 ], [ %t4i620, %forbodyi626 ]		; <i32> [#uses=2]
	%p.0i619 = phi i8* [ getelementptr ([11 x i8]* @.str170, i32 0, i32 -1), %Dt3net3ftp9FClient13FConnection13pListLineMFAaZS5t3net3ftp9FClient11FFileInfo10p_wordMFZAa.exit828 ], [ %t7i623, %forbodyi626 ]		; <i8*> [#uses=1]
	%t4i620 = add i32 %l.0i618, -1		; <i32> [#uses=1]
	%t5i621 = icmp eq i32 %l.0i618, 0		; <i1> [#uses=1]
	br i1 %t5i621, label %if65, label %forbodyi626

forbodyi626:		; preds = %forcondi622
	%t7i623 = getelementptr i8* %p.0i619, i32 1		; <i8*> [#uses=3]
	%t8i624 = load i8* %t7i623		; <i8> [#uses=1]
	%t12i625 = icmp eq i8 %t8i624, %t63		; <i1> [#uses=1]
	br i1 %t12i625, label %ifi630, label %forcondi622

ifi630:		; preds = %forbodyi626
	%t15i627 = ptrtoint i8* %t7i623 to i32		; <i32> [#uses=1]
	%t17i629 = sub i32 %t15i627, ptrtoint ([11 x i8]* @.str170 to i32)		; <i32> [#uses=1]
	%phit636 = icmp eq i32 %t17i629, 10		; <i1> [#uses=1]
	br i1 %phit636, label %if65, label %e67

if65:		; preds = %ifi630, %forcondi622
	%t4i532 = icmp eq i32 %t49.i759, %.frame.0.3		; <i1> [#uses=1]
	br i1 %t4i532, label %Dt4x7c7I11V4tTaZ4tFAaKbKkZk.exit.i576, label %forcondi539

forcondi539:		; preds = %zi546, %if65
	%sign.1.i533 = phi i1 [ %sign.0.i543, %zi546 ], [ false, %if65 ]		; <i1> [#uses=2]
	%l.0i534 = phi i32 [ %t33i545, %zi546 ], [ %t90.i783, %if65 ]		; <i32> [#uses=3]
	%p.0i535 = phi i8* [ %t30i544, %zi546 ], [ %t89.i782, %if65 ]		; <i8*> [#uses=6]
	%c.0.ini536 = phi i8* [ %t30i544, %zi546 ], [ %t89.i782, %if65 ]		; <i8*> [#uses=1]
	%c.0i537 = load i8* %c.0.ini536		; <i8> [#uses=2]
	%t8i538 = icmp eq i32 %l.0i534, 0		; <i1> [#uses=1]
	br i1 %t8i538, label %endfori550, label %forbodyi540

forbodyi540:		; preds = %forcondi539
	switch i8 %c.0i537, label %endfori550 [
		i8 32, label %zi546
		i8 9, label %zi546
		i8 45, label %if20i541
		i8 43, label %if26i542
	]

if20i541:		; preds = %forbodyi540
	br label %zi546

if26i542:		; preds = %forbodyi540
	br label %zi546

zi546:		; preds = %if26i542, %if20i541, %forbodyi540, %forbodyi540
	%sign.0.i543 = phi i1 [ false, %if26i542 ], [ true, %if20i541 ], [ %sign.1.i533, %forbodyi540 ], [ %sign.1.i533, %forbodyi540 ]		; <i1> [#uses=1]
	%t30i544 = getelementptr i8* %p.0i535, i32 1		; <i8*> [#uses=2]
	%t33i545 = add i32 %l.0i534, -1		; <i32> [#uses=1]
	br label %forcondi539

endfori550:		; preds = %forbodyi540, %forcondi539
	%t37i547 = icmp eq i8 %c.0i537, 48		; <i1> [#uses=1]
	%t39i548 = icmp sgt i32 %l.0i534, 1		; <i1> [#uses=1]
	%or.condi549 = and i1 %t37i547, %t39i548		; <i1> [#uses=1]
	br i1 %or.condi549, label %if40i554, label %endif41i564

if40i554:		; preds = %endfori550
	%t43i551 = getelementptr i8* %p.0i535, i32 1		; <i8*> [#uses=2]
	%t44i552 = load i8* %t43i551		; <i8> [#uses=1]
	%t45i553 = zext i8 %t44i552 to i32		; <i32> [#uses=1]
	switch i32 %t45i553, label %endif41i564 [
		i32 120, label %case46i556
		i32 88, label %case46i556
		i32 98, label %case51i558
		i32 66, label %case51i558
		i32 111, label %case56i560
		i32 79, label %case56i560
	]

case46i556:		; preds = %if40i554, %if40i554
	%t48i555 = getelementptr i8* %p.0i535, i32 2		; <i8*> [#uses=1]
	br label %endif41i564

case51i558:		; preds = %if40i554, %if40i554
	%t53i557 = getelementptr i8* %p.0i535, i32 2		; <i8*> [#uses=1]
	br label %endif41i564

case56i560:		; preds = %if40i554, %if40i554
	%t58i559 = getelementptr i8* %p.0i535, i32 2		; <i8*> [#uses=1]
	br label %endif41i564

endif41i564:		; preds = %case56i560, %case51i558, %case46i556, %if40i554, %endfori550
	%r.0i561 = phi i32 [ 0, %if40i554 ], [ 8, %case56i560 ], [ 2, %case51i558 ], [ 16, %case46i556 ], [ 0, %endfori550 ]		; <i32> [#uses=2]
	%p.2i562 = phi i8* [ %t43i551, %if40i554 ], [ %t58i559, %case56i560 ], [ %t53i557, %case51i558 ], [ %t48i555, %case46i556 ], [ %p.0i535, %endfori550 ]		; <i8*> [#uses=2]
	%t63i563 = icmp eq i32 %r.0i561, 0		; <i1> [#uses=1]
	br i1 %t63i563, label %Dt4x7c7I11V4tTaZ4tFAaKbKkZk.exit.i576, label %if70i568

if70i568:		; preds = %endif41i564
	br label %Dt4x7c7I11V4tTaZ4tFAaKbKkZk.exit.i576

Dt4x7c7I11V4tTaZ4tFAaKbKkZk.exit.i576:		; preds = %if70i568, %endif41i564, %if65
	%radix.0.i570 = phi i32 [ 0, %if65 ], [ %r.0i561, %if70i568 ], [ 10, %endif41i564 ]		; <i32> [#uses=2]
	%p.1i571 = phi i8* [ %p.2i562, %if70i568 ], [ %t89.i782, %if65 ], [ %p.2i562, %endif41i564 ]		; <i8*> [#uses=1]
	%t84i572 = ptrtoint i8* %p.1i571 to i32		; <i32> [#uses=1]
	%t85i573 = ptrtoint i8* %t89.i782 to i32		; <i32> [#uses=1]
	%t86i574 = sub i32 %t84i572, %t85i573		; <i32> [#uses=2]
	%t6.i575 = sub i32 %t90.i783, %t86i574		; <i32> [#uses=1]
	%t59i604 = zext i32 %radix.0.i570 to i64		; <i64> [#uses=1]
	br label %fcondi581

fcondi581:		; preds = %if55i610, %Dt4x7c7I11V4tTaZ4tFAaKbKkZk.exit.i576
	%value.0i577 = phi i64 [ 0, %Dt4x7c7I11V4tTaZ4tFAaKbKkZk.exit.i576 ], [ %t65i607, %if55i610 ]		; <i64> [#uses=1]
	%fkey.0i579 = phi i32 [ 0, %Dt4x7c7I11V4tTaZ4tFAaKbKkZk.exit.i576 ], [ %t70i609, %if55i610 ]		; <i32> [#uses=3]
	%t3i580 = icmp ult i32 %fkey.0i579, %t6.i575		; <i1> [#uses=1]
	br i1 %t3i580, label %fbodyi587, label %wcond.i422

fbodyi587:		; preds = %fcondi581
	%t5.s.i582 = add i32 %t86i574, %fkey.0i579		; <i32> [#uses=1]
	%t89.i782.s = add i32 %.frame.0.3, %t5.s.i582		; <i32> [#uses=1]
	%t5i583 = getelementptr i8* %line_arg831, i32 %t89.i782.s		; <i8*> [#uses=1]
	%t6i584 = load i8* %t5i583		; <i8> [#uses=6]
	%t6.off84i585 = add i8 %t6i584, -48		; <i8> [#uses=1]
	%or.cond.i28.i586 = icmp ugt i8 %t6.off84i585, 9		; <i1> [#uses=1]
	br i1 %or.cond.i28.i586, label %ei590, label %endifi603

ei590:		; preds = %fbodyi587
	%t6.off83i588 = add i8 %t6i584, -97		; <i8> [#uses=1]
	%or.cond81i589 = icmp ugt i8 %t6.off83i588, 25		; <i1> [#uses=1]
	br i1 %or.cond81i589, label %e24i595, label %if22i592

if22i592:		; preds = %ei590
	%t27i591 = add i8 %t6i584, -39		; <i8> [#uses=1]
	br label %endifi603

e24i595:		; preds = %ei590
	%t6.offi593 = add i8 %t6i584, -65		; <i8> [#uses=1]
	%or.cond82i594 = icmp ugt i8 %t6.offi593, 25		; <i1> [#uses=1]
	br i1 %or.cond82i594, label %wcond.i422, label %if39i597

if39i597:		; preds = %e24i595
	%t44.i29.i596 = add i8 %t6i584, -7		; <i8> [#uses=1]
	br label %endifi603

endifi603:		; preds = %if39i597, %if22i592, %fbodyi587
	%c.0.i30.i598 = phi i8 [ %t27i591, %if22i592 ], [ %t44.i29.i596, %if39i597 ], [ %t6i584, %fbodyi587 ]		; <i8> [#uses=1]
	%t48.i31.i599 = zext i8 %c.0.i30.i598 to i32		; <i32> [#uses=1]
	%t49i600 = add i32 %t48.i31.i599, 208		; <i32> [#uses=1]
	%t52i601 = and i32 %t49i600, 255		; <i32> [#uses=2]
	%t54i602 = icmp ult i32 %t52i601, %radix.0.i570		; <i1> [#uses=1]
	br i1 %t54i602, label %if55i610, label %wcond.i422

if55i610:		; preds = %endifi603
	%t61i605 = mul i64 %value.0i577, %t59i604		; <i64> [#uses=1]
	%t64i606 = zext i32 %t52i601 to i64		; <i64> [#uses=1]
	%t65i607 = add i64 %t61i605, %t64i606		; <i64> [#uses=1]
	%t70i609 = add i32 %fkey.0i579, 1		; <i32> [#uses=1]
	br label %fcondi581

e67:		; preds = %ifi630
	%t4i447 = icmp eq i32 %t49.i801, %.frame.0.6		; <i1> [#uses=1]
	br i1 %t4i447, label %Dt4x7c7I11V4tTaZ4tFAaKbKkZk.exit.i491, label %forcondi454

forcondi454:		; preds = %zi461, %e67
	%c.0i452 = phi i8 [ %c.0i452.pre, %zi461 ], [ %t63, %e67 ]		; <i8> [#uses=2]
	%sign.1.i448 = phi i1 [ %sign.0.i458, %zi461 ], [ false, %e67 ]		; <i1> [#uses=2]
	%l.0i449 = phi i32 [ %t33i460, %zi461 ], [ %t90.i825, %e67 ]		; <i32> [#uses=3]
	%p.0i450 = phi i8* [ %t30i459, %zi461 ], [ %t89.i824, %e67 ]		; <i8*> [#uses=5]
	%t8i453 = icmp eq i32 %l.0i449, 0		; <i1> [#uses=1]
	br i1 %t8i453, label %endfori465, label %forbodyi455

forbodyi455:		; preds = %forcondi454
	switch i8 %c.0i452, label %endfori465 [
		i8 32, label %zi461
		i8 9, label %zi461
		i8 45, label %if20i456
		i8 43, label %if26i457
	]

if20i456:		; preds = %forbodyi455
	br label %zi461

if26i457:		; preds = %forbodyi455
	br label %zi461

zi461:		; preds = %if26i457, %if20i456, %forbodyi455, %forbodyi455
	%sign.0.i458 = phi i1 [ false, %if26i457 ], [ true, %if20i456 ], [ %sign.1.i448, %forbodyi455 ], [ %sign.1.i448, %forbodyi455 ]		; <i1> [#uses=1]
	%t30i459 = getelementptr i8* %p.0i450, i32 1		; <i8*> [#uses=2]
	%t33i460 = add i32 %l.0i449, -1		; <i32> [#uses=1]
	%c.0i452.pre = load i8* %t30i459		; <i8> [#uses=1]
	br label %forcondi454

endfori465:		; preds = %forbodyi455, %forcondi454
	%t37i462 = icmp eq i8 %c.0i452, 48		; <i1> [#uses=1]
	%t39i463 = icmp sgt i32 %l.0i449, 1		; <i1> [#uses=1]
	%or.condi464 = and i1 %t37i462, %t39i463		; <i1> [#uses=1]
	br i1 %or.condi464, label %if40i469, label %endif41i479

if40i469:		; preds = %endfori465
	%t43i466 = getelementptr i8* %p.0i450, i32 1		; <i8*> [#uses=2]
	%t44i467 = load i8* %t43i466		; <i8> [#uses=1]
	%t45i468 = zext i8 %t44i467 to i32		; <i32> [#uses=1]
	switch i32 %t45i468, label %endif41i479 [
		i32 120, label %case46i471
		i32 111, label %case56i475
	]

case46i471:		; preds = %if40i469
	%t48i470 = getelementptr i8* %p.0i450, i32 2		; <i8*> [#uses=1]
	br label %endif41i479

case56i475:		; preds = %if40i469
	%t58i474 = getelementptr i8* %p.0i450, i32 2		; <i8*> [#uses=1]
	br label %endif41i479

endif41i479:		; preds = %case56i475, %case46i471, %if40i469, %endfori465
	%r.0i476 = phi i32 [ 0, %if40i469 ], [ 8, %case56i475 ], [ 16, %case46i471 ], [ 0, %endfori465 ]		; <i32> [#uses=2]
	%p.2i477 = phi i8* [ %t43i466, %if40i469 ], [ %t58i474, %case56i475 ], [ %t48i470, %case46i471 ], [ %p.0i450, %endfori465 ]		; <i8*> [#uses=2]
	%t63i478 = icmp eq i32 %r.0i476, 0		; <i1> [#uses=1]
	br i1 %t63i478, label %Dt4x7c7I11V4tTaZ4tFAaKbKkZk.exit.i491, label %if70i483

if70i483:		; preds = %endif41i479
	br label %Dt4x7c7I11V4tTaZ4tFAaKbKkZk.exit.i491

Dt4x7c7I11V4tTaZ4tFAaKbKkZk.exit.i491:		; preds = %if70i483, %endif41i479, %e67
	%radix.0.i485 = phi i32 [ 0, %e67 ], [ %r.0i476, %if70i483 ], [ 10, %endif41i479 ]		; <i32> [#uses=2]
	%p.1i486 = phi i8* [ %p.2i477, %if70i483 ], [ %t89.i824, %e67 ], [ %p.2i477, %endif41i479 ]		; <i8*> [#uses=1]
	%t84i487 = ptrtoint i8* %p.1i486 to i32		; <i32> [#uses=1]
	%t85i488 = ptrtoint i8* %t89.i824 to i32		; <i32> [#uses=1]
	%t86i489 = sub i32 %t84i487, %t85i488		; <i32> [#uses=2]
	%ttt = sub i32 %t90.i825, %t86i489		; <i32> [#uses=1]
	%t59i519 = zext i32 %radix.0.i485 to i64		; <i64> [#uses=1]
	br label %fcondi496

fcondi496:		; preds = %if55i525, %Dt4x7c7I11V4tTaZ4tFAaKbKkZk.exit.i491
	%value.0i492 = phi i64 [ 0, %Dt4x7c7I11V4tTaZ4tFAaKbKkZk.exit.i491 ], [ %t65i522, %if55i525 ]		; <i64> [#uses=1]
	%fkey.0i494 = phi i32 [ 0, %Dt4x7c7I11V4tTaZ4tFAaKbKkZk.exit.i491 ], [ %t70i524, %if55i525 ]		; <i32> [#uses=3]
	%t3i495 = icmp ult i32 %fkey.0i494, %ttt		; <i1> [#uses=1]
	br i1 %t3i495, label %fbodyi502, label %wcond.i422

fbodyi502:		; preds = %fcondi496
	%t5.s.i497 = add i32 %t86i489, %fkey.0i494		; <i32> [#uses=1]
	%t89.i824.s = add i32 %.frame.0.6, %t5.s.i497		; <i32> [#uses=1]
	%t5i498 = getelementptr i8* %line_arg831, i32 %t89.i824.s		; <i8*> [#uses=1]
	%t6i499 = load i8* %t5i498		; <i8> [#uses=6]
	%t6.off84i500 = add i8 %t6i499, -48		; <i8> [#uses=1]
	%or.cond.i28.i501 = icmp ugt i8 %t6.off84i500, 9		; <i1> [#uses=1]
	br i1 %or.cond.i28.i501, label %ei505, label %endifi518

ei505:		; preds = %fbodyi502
	%t6.off83i503 = add i8 %t6i499, -97		; <i8> [#uses=1]
	%or.cond81i504 = icmp ugt i8 %t6.off83i503, 25		; <i1> [#uses=1]
	br i1 %or.cond81i504, label %e24i510, label %if22i507

if22i507:		; preds = %ei505
	%t27i506 = add i8 %t6i499, -39		; <i8> [#uses=1]
	br label %endifi518

e24i510:		; preds = %ei505
	%t6.offi508 = add i8 %t6i499, -65		; <i8> [#uses=1]
	%or.cond82i509 = icmp ugt i8 %t6.offi508, 25		; <i1> [#uses=1]
	br i1 %or.cond82i509, label %wcond.i422, label %if39i512

if39i512:		; preds = %e24i510
	%t44.i29.i511 = add i8 %t6i499, -7		; <i8> [#uses=1]
	br label %endifi518

endifi518:		; preds = %if39i512, %if22i507, %fbodyi502
	%c.0.i30.i513 = phi i8 [ %t27i506, %if22i507 ], [ %t44.i29.i511, %if39i512 ], [ %t6i499, %fbodyi502 ]		; <i8> [#uses=1]
	%t48.i31.i514 = zext i8 %c.0.i30.i513 to i32		; <i32> [#uses=1]
	%t49i515 = add i32 %t48.i31.i514, 208		; <i32> [#uses=1]
	%t52i516 = and i32 %t49i515, 255		; <i32> [#uses=2]
	%t54i517 = icmp ult i32 %t52i516, %radix.0.i485		; <i1> [#uses=1]
	br i1 %t54i517, label %if55i525, label %wcond.i422

if55i525:		; preds = %endifi518
	%t61i520 = mul i64 %value.0i492, %t59i519		; <i64> [#uses=1]
	%t64i521 = zext i32 %t52i516 to i64		; <i64> [#uses=1]
	%t65i522 = add i64 %t61i520, %t64i521		; <i64> [#uses=1]
	%t70i524 = add i32 %fkey.0i494, 1		; <i32> [#uses=1]
	br label %fcondi496

wcond.i422:		; preds = %e40.i, %endifi518, %e24i510, %fcondi496, %endifi603, %e24i595, %fcondi581
	%sarg60.pn.i = phi i8* [ %p.0.i, %e40.i ], [ undef, %fcondi496 ], [ undef, %e24i510 ], [ undef, %endifi518 ], [ undef, %endifi603 ], [ undef, %e24i595 ], [ undef, %fcondi581 ]		; <i8*> [#uses=3]
	%start_arg.pn.i = phi i32 [ %t49.i443, %e40.i ], [ 0, %fcondi496 ], [ 0, %e24i510 ], [ 0, %endifi518 ], [ 0, %endifi603 ], [ 0, %e24i595 ], [ 0, %fcondi581 ]		; <i32> [#uses=3]
	%extent.0.i = phi i32 [ %t51.i, %e40.i ], [ undef, %fcondi496 ], [ undef, %e24i510 ], [ undef, %endifi518 ], [ undef, %endifi603 ], [ undef, %e24i595 ], [ undef, %fcondi581 ]		; <i32> [#uses=3]
	%p.0.i = getelementptr i8* %sarg60.pn.i, i32 %start_arg.pn.i		; <i8*> [#uses=2]
	%p.0.s63.i = add i32 %start_arg.pn.i, -1		; <i32> [#uses=1]
	%t2i424 = getelementptr i8* %sarg60.pn.i, i32 %p.0.s63.i		; <i8*> [#uses=1]
	br label %forcondi430

forcondi430:		; preds = %forbodyi434, %wcond.i422
	%l.0i426 = phi i32 [ %extent.0.i, %wcond.i422 ], [ %t4i428, %forbodyi434 ]		; <i32> [#uses=2]
	%p.0i427 = phi i8* [ %t2i424, %wcond.i422 ], [ %t7i431, %forbodyi434 ]		; <i8*> [#uses=1]
	%t4i428 = add i32 %l.0i426, -1		; <i32> [#uses=1]
	%t5i429 = icmp eq i32 %l.0i426, 0		; <i1> [#uses=1]
	br i1 %t5i429, label %e.i441, label %forbodyi434

forbodyi434:		; preds = %forcondi430
	%t7i431 = getelementptr i8* %p.0i427, i32 1		; <i8*> [#uses=3]
	%t8i432 = load i8* %t7i431		; <i8> [#uses=1]
	%t12i433 = icmp eq i8 %t8i432, 32		; <i1> [#uses=1]
	br i1 %t12i433, label %ifi438, label %forcondi430

ifi438:		; preds = %forbodyi434
	%t15i435 = ptrtoint i8* %t7i431 to i32		; <i32> [#uses=1]
	%t16i436 = ptrtoint i8* %p.0.i to i32		; <i32> [#uses=1]
	%t17i437 = sub i32 %t15i435, %t16i436		; <i32> [#uses=1]
	br label %e.i441

e.i441:		; preds = %ifi438, %forcondi430
	%t2561.i = phi i32 [ %t17i437, %ifi438 ], [ %extent.0.i, %forcondi430 ]		; <i32> [#uses=2]
	%p.0.s.i = add i32 %start_arg.pn.i, %t2561.i		; <i32> [#uses=1]
	%t32.s.i = add i32 %p.0.s.i, -1		; <i32> [#uses=1]
	%t2i.i = getelementptr i8* %sarg60.pn.i, i32 %t32.s.i		; <i8*> [#uses=1]
	br label %forbodyi.i

forbodyi.i:		; preds = %forbodyi.i, %e.i441
	%p.0i.i = phi i8* [ %t2i.i, %e.i441 ], [ %t7i.i, %forbodyi.i ]		; <i8*> [#uses=1]
	%s2.0i.i = phi i8* [ getelementptr ([5 x i8]* @.str171, i32 0, i32 0), %e.i441 ], [ %t11i.i, %forbodyi.i ]		; <i8*> [#uses=2]
	%t7i.i = getelementptr i8* %p.0i.i, i32 1		; <i8*> [#uses=2]
	%t8i.i = load i8* %t7i.i		; <i8> [#uses=1]
	%t11i.i = getelementptr i8* %s2.0i.i, i32 1		; <i8*> [#uses=1]
	%t12i.i = load i8* %s2.0i.i		; <i8> [#uses=1]
	%t14i.i = icmp eq i8 %t8i.i, %t12i.i		; <i1> [#uses=1]
	br i1 %t14i.i, label %forbodyi.i, label %e40.i

e40.i:		; preds = %forbodyi.i
	%t49.i443 = add i32 %t2561.i, 1		; <i32> [#uses=2]
	%t51.i = sub i32 %extent.0.i, %t49.i443		; <i32> [#uses=1]
	br label %wcond.i422
}
