void *dlclose(void*);

void ap_os_dso_unload(void *handle)
{
    dlclose(handle);
    return;     /* This return triggers the bug: Wierd */
}
