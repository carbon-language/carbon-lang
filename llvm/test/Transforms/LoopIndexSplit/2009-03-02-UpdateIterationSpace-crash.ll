; RUN: opt < %s -loop-index-split -disable-output
	%struct.CGPoint = type { double, double }
	%struct.IBCFMutableDictionary = type { %struct.NSMutableArray, %struct.__CFDictionary*, %struct.NSSortDescriptor*, %struct.NSSortDescriptor* }
	%struct.IBInspectorMode = type opaque
	%struct.IBInspectorModeView = type { %struct.NSView, %struct.NSArray*, %struct.IBCFMutableDictionary*, %struct.IBInspectorMode*, %struct.IBInspectorMode*, %struct.IBInspectorMode*, %struct.objc_selector*, %struct.NSObject* }
	%struct.NSArray = type { %struct.NSObject }
	%struct.NSImage = type { %struct.NSObject, %struct.NSArray*, %struct.CGPoint, %struct.__imageFlags, %struct.NSObject*, %struct._NSImageAuxiliary* }
	%struct.NSMutableArray = type { %struct.NSArray }
	%struct.NSObject = type { %struct.objc_class* }
	%struct.NSRect = type { %struct.CGPoint, %struct.CGPoint }
	%struct.NSResponder = type { %struct.NSObject, %struct.NSObject* }
	%struct.NSSortDescriptor = type { %struct.NSObject, i64, %struct.NSArray*, %struct.objc_selector*, %struct.NSObject* }
	%struct.NSURL = type { %struct.NSObject, %struct.NSArray*, %struct.NSURL*, i8*, i8* }
	%struct.NSView = type { %struct.NSResponder, %struct.NSRect, %struct.NSRect, %struct.NSObject*, %struct.NSObject*, %struct.NSWindow*, %struct.NSObject*, %struct.NSObject*, %struct.NSObject*, %struct.NSObject*, %struct._NSViewAuxiliary*, %struct._VFlags, %struct.__VFlags2 }
	%struct.NSWindow = type { %struct.NSResponder, %struct.NSRect, %struct.NSObject*, %struct.NSObject*, %struct.NSResponder*, %struct.NSView*, %struct.NSView*, %struct.NSObject*, %struct.NSObject*, i32, i64, i32, %struct.NSArray*, %struct.NSObject*, i8, i8, i8, i8, i8*, i8*, %struct.NSImage*, i32, %struct.NSMutableArray*, %struct.NSURL*, %struct.CGPoint*, %struct.NSArray*, %struct.NSArray*, %struct.__wFlags, %struct.NSObject*, %struct.NSView*, %struct.NSWindowAuxiliary* }
	%struct.NSWindowAuxiliary = type opaque
	%struct._NSImageAuxiliary = type opaque
	%struct._NSViewAuxiliary = type opaque
	%struct._VFlags = type <{ i8, i8, i8, i8 }>
	%struct.__CFDictionary = type opaque
	%struct.__VFlags2 = type <{ i32 }>
	%struct.__imageFlags = type <{ i8, [3 x i8] }>
	%struct.__wFlags = type <{ i8, i8, i8, i8, i8, i8, i8, i8 }>
	%struct.objc_class = type opaque
	%struct.objc_selector = type opaque

define %struct.NSArray* @"\01-[IBInspectorModeView calculateModeRects]"(%struct.IBInspectorModeView* %self, %struct.objc_selector* %_cmd) optsize ssp {
entry:
	br i1 false, label %bb7, label %bb

bb:		; preds = %entry
	br i1 false, label %bb.nph, label %bb7.loopexit

bb.nph:		; preds = %bb
	br label %bb1

bb1:		; preds = %bb6, %bb.nph
	%midx.01 = phi i64 [ %3, %bb6 ], [ 0, %bb.nph ]		; <i64> [#uses=3]
	%0 = icmp sge i64 %midx.01, 0		; <i1> [#uses=1]
	%1 = icmp sle i64 %midx.01, 0		; <i1> [#uses=1]
	%2 = and i1 %0, %1		; <i1> [#uses=1]
	br i1 %2, label %bb4, label %bb5

bb4:		; preds = %bb1
	br label %bb5

bb5:		; preds = %bb4, %bb1
	%modeWidth.0 = phi double [ 0.000000e+00, %bb1 ], [ 0.000000e+00, %bb4 ]		; <double> [#uses=0]
	%3 = add i64 %midx.01, 1		; <i64> [#uses=1]
	br label %bb6

bb6:		; preds = %bb5
	%4 = icmp slt i64 0, 0		; <i1> [#uses=1]
	br i1 %4, label %bb1, label %bb6.bb7.loopexit_crit_edge

bb6.bb7.loopexit_crit_edge:		; preds = %bb6
	br label %bb7.loopexit

bb7.loopexit:		; preds = %bb6.bb7.loopexit_crit_edge, %bb
	br label %bb7

bb7:		; preds = %bb7.loopexit, %entry
	ret %struct.NSArray* null
}
