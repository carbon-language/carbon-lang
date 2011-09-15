const char *getModuleVersion(void);

#ifdef FOO
#  error Module should have been built without -DFOO
#endif

@interface Module
+(const char *)version; // retrieve module version
+alloc;
@end

