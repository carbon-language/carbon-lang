typedef struct {
    char *auth_pwfile;
    int x;
} auth_config_rec;

void *Ptr = &((auth_config_rec*)0)->x;

int main() { return 0; }
