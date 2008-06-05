// RUN: clang -fsyntax-only -verify %s
typedef struct objc_selector *SEL;
typedef signed char BOOL;
typedef int NSInteger;
typedef unsigned int NSUInteger;
typedef struct _NSZone NSZone;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
@protocol NSObject
- (BOOL)isEqual:(id)object;
- (BOOL)conformsToProtocol:(Protocol *)aProtocol;
@end

@protocol NSCopying
- (id)copyWithZone:(NSZone *)zone;
@end

@protocol NSMutableCopying
- (id)mutableCopyWithZone:(NSZone *)zone;
@end

@protocol NSCoding
- (void)encodeWithCoder:(NSCoder *)aCoder;
@end

@interface NSObject <NSObject> {}

- (void)dealloc;
@end

extern id NSAllocateObject(Class aClass, NSUInteger extraBytes, NSZone *zone);

typedef struct _NSSize {} NSRect;
typedef struct {} NSFastEnumerationState;

@protocol NSFastEnumeration
- (NSUInteger)countByEnumeratingWithState:(NSFastEnumerationState *)state objects:(id *)stackbuf count:(NSUInteger)len;
@end

@class NSString;
@interface NSString : NSObject <NSCopying, NSMutableCopying, NSCoding>  
- (NSUInteger)length;
- (BOOL)isEqualToString:(NSString *)aString;
@end

@interface NSSimpleCString : NSString {} @end

@interface NSConstantString : NSSimpleCString @end

extern void *_NSConstantStringClassReference;

@interface NSSet : NSObject <NSCopying, NSMutableCopying, NSCoding, NSFastEnumeration>
- (NSUInteger)count;
@end

@interface NSMutableSet : NSSet
- (void)addObject:(id)object;
@end

@class NSArray, NSDictionary, NSMapTable;
@interface NSResponder : NSObject <NSCoding> {} @end

@protocol NSAnimatablePropertyContainer    
- (id)animator;
@end

extern NSString *NSAnimationTriggerOrderIn ;
@interface NSView : NSResponder  <NSAnimatablePropertyContainer> {} @end

extern NSString * const NSFullScreenModeAllScreens;
typedef NSUInteger NSControlTint;

@interface NSCell : NSObject <NSCopying, NSCoding> {}
- (NSRect)imageRectForBounds:(NSRect)theRect;
@end

@protocol NSValidatedUserInterfaceItem
- (SEL)action;
@end

@protocol NSUserInterfaceValidations
- (BOOL)validateUserInterfaceItem:(id <NSValidatedUserInterfaceItem>)anItem;
@end

@class NSCell, NSFont, NSTextView, NSNotification, NSAttributedString, NSFormatter;
@interface NSControl : NSView {} @end

@class NSColor, NSClipView, NSRulerView, NSScroller;
@interface NSWindowController : NSResponder <NSCoding> {} @end

@class NSTableHeaderView;
@class NSTableColumn;
@interface NSTableView : NSControl <NSUserInterfaceValidations> {}

- (NSInteger)columnWithIdentifier:(id)identifier;
- (NSRect)frameOfCellAtColumn:(NSInteger)column row:(NSInteger)row;
@end

@class NSButtonCell;
@interface NSOutlineView : NSTableView {}
- (NSInteger)rowForItem:(id)item;
@end

@class NSArray, NSDictionary, NSMutableArray, NSNotification, NSString, NSToolbarItem, NSWindow;
@protocol XCProxyObjectProtocol
- (id) representedObject;
@end

@interface PBXObject : NSObject {} @end
typedef enum
{
    PBXNoItemChanged = 0x00,     PBXProjectItemChanged = 0x01,     PBXReferenceChanged = 0x02,     PBXGroupChanged = 0x04,     PBXTargetChanged = 0x08,     PBXBuildPhaseChanged = 0x10,     PBXBuildFileChanged = 0x20,     PBXBreakpointChanged = 0x40,
}

PBXChangedItemMask;
@protocol PBXChangeNotification 
- (void)willChange;
@end

@class PBXContainer, PBXProject;
@interface PBXContainerItem : PBXObject <PBXChangeNotification> {} @end

@interface PBXProjectItem : PBXContainerItem {} @end

@class XCObjectGraphPath;
@protocol XCCompatibilityChecking  
- (void)findFeaturesInUseAndAddToSet:(NSMutableSet *)featureSet usingPathPrefix:(XCObjectGraphPath *)pathPrefix;
- (NSString *)identifier;
@end

@protocol XCConfigurationInspectables <NSObject> 
- (NSString *)name;
@end

@class PBXProject, PBXFileReference, PBXBuildPhase, PBXBuildSettingsDictionary, PBXExecutable, PBXBuildFile, PBXTargetDependency, PBXBuildLog, PBXBuildRule, XCCommandLineToolSpecification, XCProductTypeSpecification, PBXPackageTypeSpecification, PBXTargetBuildContext, XCBuildConfiguration, XCConfigurationList, XCHeadersBuildPhaseDGSnapshot, XCResourcesBuildPhaseDGSnapshot, XCSourcesBuildPhaseDGSnapshot, XCFrameworksBuildPhaseDGSnapshot, XCRezBuildPhaseDGSnapshot, XCJavaArchiveBuildPhaseDGSnapshot, XCBuildFileRefDGSnapshot, XCWorkQueue, XCBuildOperation, XCStringList, XCPropertyExpansionContext, XCWorkQueueOperation, XCTargetDGSnapshot, XCTargetHeadermapCreationInfo, XCPropertyInfoContext, XCConfigurationInspectionContext, PBXReference;
@interface PBXTarget : PBXProjectItem <XCCompatibilityChecking, XCConfigurationInspectables> {} @end

extern NSString * const XCTargetDGSnapshotContextKey_BuildAction;
@interface PBXBookmarkItem : PBXProjectItem {} @end

@interface PBXReference : PBXContainerItem {} @end

extern BOOL PBX_getUsesTabsPreference();
@interface PBXGroup : PBXReference <XCCompatibilityChecking> {} @end

@class PBXFileReference, PBXTarget, PBXProject;
@interface PBXExecutable : PBXProjectItem {} @end

@class XCSCMRevisionInfo;
@interface PBXBookmark : PBXBookmarkItem {} @end

@class XCSCMInfo;
@interface PBXFileReference : PBXReference {} @end

@interface PBXLegacyTarget : PBXTarget {} @end

@interface PBXVariantGroup : PBXGroup {} @end

typedef enum
{
    PBXBuildMessageType_None,     PBXBuildMessageType_Notice,     PBXBuildMessageType_Warning,     PBXBuildMessageType_Error,
}

PBXBuildMessageType;
@interface PBXBuildMessage : NSObject {} @end

@class PBXBreakpoint, PBXFileReference, PBXProject, PBXTextBookmark;
@protocol PBXMarkerDelegateProtocol <NSObject>
- (void) setLineNumber:(NSUInteger)newLineNumber;
@end

typedef enum
{
    PBXBreakpointIgnoreCountType = 0,  PBXBreakpointMultipleCountType
}

PBXBreakpointCountType;

@interface PBXBreakpoint : PBXProjectItem {} @end
@interface PBXFileBreakpoint : PBXBreakpoint <NSCopying, PBXMarkerDelegateProtocol> {} @end

extern NSString *XCBreakpointActionsWereUpdated;
@protocol PBXNodeEditingProtocol
- (BOOL) canRename;
@end

@protocol XCFosterParentHostProtocol
- (void) reloadDataForProxies;
@end

@interface PBXBuildLogItem : NSObject {} @end

@interface PBXBuildLogMessageItem : PBXBuildLogItem {} @end

extern NSString *PBXWindowDidChangeFirstResponderNotification;
@interface PBXModule : NSWindowController {} @end
typedef enum
{
    PBXPanelCanChooseFiles,     PBXPanelCanChooseFolders,     PBXPanelCanChooseBoth,     PBXPanelCanChooseOnlyExistingFolders
}

PBXPanelSelection;
@interface XCSelection : NSResponder {} @end

@protocol XCSelectionSource
- (XCSelection *) xcSelection;
@end
typedef enum
{
    PBXFindMatchContains,     PBXFindMatchStartsWith,     PBXFindMatchWholeWords,     PBXFindMatchEndsWith
}

PBXFindMatchStyle;
@protocol PBXSelectableText
- (NSString *)selectedString;
@end

@protocol PBXFindableText <PBXSelectableText>  
- (BOOL)findText:(NSString *)string ignoreCase:(BOOL)ignoreCase matchStyle:(PBXFindMatchStyle)matchStyle backwards:(BOOL)backwards wrap:(BOOL)wrap;
@end

@class PBXProjectDocument, PBXProject, PBXAttributedStatusView;
@interface PBXProjectModule : PBXModule <PBXFindableText> {} @end

@class PBXExtendedOutlineView, PBXFileReference, PBXGroup, PBXProject, PBXProjectDocument, PBXReference, PBXOutlineDataSourceSplitter, XCSCMInfo;
extern NSString * const PBXGroupTreeMainColumnIdentifier;
@interface PBXGroupTreeModule : PBXProjectModule {} @end

@protocol PBXTableColumnProvider  
- (NSArray *) optionalColumnIdentifiers:(NSTableView *)tableView;
@end

extern NSString *PBXSmartGroupTreeModuleColumnsKey;
@interface PBXSmartGroupTreeModule : PBXGroupTreeModule <PBXTableColumnProvider, XCSelectionSource, XCFosterParentHostProtocol> {} @end

@class PBXBookmark, PBXProjectModule, PBXProjectDocument, PBXSmartGroupTreeModule, PBXBreakpoint, XCBreakpointsBucket, PBXFileNavigator;
@interface PBXFosterParent : PBXGroup <XCProxyObjectProtocol, PBXNodeEditingProtocol> {} @end

@class NSString, NSAttributedString, PBXBookmark, PBXFileDocument, PBXSymbol, PBXDocBookmark;
@interface PBXFindResult : NSObject {} @end

@protocol PBXBookmarkSupport
- (PBXBookmark *) bookmark;
@end

@interface PBXReference (BookmarkSupportAPI) <PBXBookmarkSupport> @end

@interface PBXBookmark (BookmarkSupportAPI) <PBXBookmarkSupport> @end

@interface PBXFileReference (BookmarkSupportAPI) <PBXBookmarkSupport> @end

@interface PBXTarget (BookmarkSupportAPI) <PBXBookmarkSupport> @end

@interface PBXLegacyTarget (BookmarkSupportAPI) <PBXBookmarkSupport> @end

@interface PBXExecutable (BookmarkSupportAPI) <PBXBookmarkSupport> @end

@interface PBXFosterParent (BookmarkSupportAPI) <PBXBookmarkSupport> @end

@interface PBXVariantGroup (BookmarkSupportAPI) <PBXBookmarkSupport> @end

@interface PBXFindResult (BookmarkSupportAPI) <PBXBookmarkSupport> @end

@interface PBXBuildMessage (BookmarkSupportAPI) <PBXBookmarkSupport> @end

@interface PBXBuildLogMessageItem (BookmarkSupportAPI) <PBXBookmarkSupport> @end

@interface PBXFileBreakpoint (BookmarkSupportAPI) <PBXBookmarkSupport> @end

extern BOOL PBXShouldIncludeReference(id ref);
@class PBXSmartGroupDataSource, PBXModule, PBXSmartGroupBinding, PBXProjectModule, PBXFosterParent, PBXExtendedOutlineView, PBXOutlineViewCell, PBXProjectWorkspaceModule;
@protocol XCOutlineViewCheckBoxProtocol
- (void) toggleEnabledState;
- (void) storeCheckBoxBounds:(NSRect)bounds;
@end

extern NSControlTint _NSDefaultControlTint(void);
@implementation PBXSmartGroupTreeModule
- (void) dealloc
{
    [super dealloc];
}

- (void)outlineView:(NSOutlineView *)outlineView willDisplayCell:(PBXOutlineViewCell *)cell forTableColumn:(NSTableColumn *)tableColumn item:(id)item
{
    if ([[tableColumn identifier] isEqualToString: PBXGroupTreeMainColumnIdentifier])
    {
        if ([item conformsToProtocol:@protocol(XCOutlineViewCheckBoxProtocol)])
        {
            NSInteger columnIndex = [outlineView columnWithIdentifier:[tableColumn identifier]];
            NSInteger row = [outlineView rowForItem:item];
            <XCOutlineViewCheckBoxProtocol> xxx;
            if (row > -1 && columnIndex > -1)
            {
                // FIXME: need to associate the correct type with this.
                [(<XCOutlineViewCheckBoxProtocol>)item storeCheckBoxBounds:[cell imageRectForBounds:[outlineView frameOfCellAtColumn:columnIndex row:row]]]; // expected-error{{bad receiver type 'int'}}
            }
        }
    }
}

