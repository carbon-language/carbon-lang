// RUN: clang -verify %s

@protocol P1 @end
@protocol P2 @end
@protocol P3 @end

@interface NSData @end

@interface MutableNSData : NSData @end

@interface Base : NSData <P1>
@property(readonly) id ref;
@property(readonly) Base *p_base;
@property(readonly) NSData *nsdata;
@property(readonly) NSData * m_nsdata;
@end

@interface Data : Base <P1, P2>
@property(readonly) NSData *ref;	// expected-warning {{property type 'NSData *' does not match property type inherited from 'Base'}}
@property(readonly) Data *p_base;	// expected-warning {{property type 'Data *' does not match property type inherited from 'Base'}}
@property(readonly) MutableNSData * m_nsdata;  // expected-warning {{property type 'MutableNSData *' does not match property type inherited from 'Base'}}
@end

@interface  MutedData: Data
@property(readonly) id p_base; // expected-warning {{property type 'id' does not match property type inherited from 'Data'}}
@end

@interface ConstData : Data <P1, P2, P3>
@property(readonly) ConstData *p_base; // expected-warning {{property type 'ConstData *' does not match property type inherited from 'Data'}}
@end

