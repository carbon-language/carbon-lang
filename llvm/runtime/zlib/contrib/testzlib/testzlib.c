
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include "zlib.h"

int ReadFileMemory(const char* filename,long* plFileSize,void** pFilePtr)
{
    FILE* stream;
    void* ptr;
    int retVal=1;
    stream=fopen(filename, "rb");
    if (stream==NULL)
        return 0;

    fseek(stream,0,SEEK_END);

    *plFileSize=ftell(stream);
    fseek(stream,0,SEEK_SET);
    ptr=malloc((*plFileSize)+1);
    if (ptr==NULL)
        retVal=0;
    else
    {
        if (fread(ptr, 1, *plFileSize,stream) != (*plFileSize))
            retVal=0;
    }
    fclose(stream);
    *pFilePtr=ptr;
    return retVal;
}

int main(int argc, char *argv[])
{
    int BlockSizeCompress=0x8000;
    int BlockSizeUncompress=0x8000;
    int cprLevel=Z_DEFAULT_COMPRESSION ;
    long lFileSize;
    unsigned char* FilePtr;
    long lBufferSizeCpr;
    long lBufferSizeUncpr;
    long lCompressedSize=0;
    unsigned char* CprPtr;
    unsigned char* UncprPtr;
    long lSizeCpr,lSizeUncpr;
    DWORD dwGetTick;

    if (argc<=1)
    {
        printf("run TestZlib <File> [BlockSizeCompress] [BlockSizeUncompress] [compres. level]\n");
        return 0;
    }

    if (ReadFileMemory(argv[1],&lFileSize,&FilePtr)==0)
    {
        printf("error reading %s\n",argv[1]);
        return 1;
    }
    else printf("file %s read, %u bytes\n",argv[1],lFileSize);

    if (argc>=3)
        BlockSizeCompress=atol(argv[2]);

    if (argc>=4)
        BlockSizeUncompress=atol(argv[3]);

    if (argc>=5)
        cprLevel=(int)atol(argv[4]);

    lBufferSizeCpr = lFileSize + (lFileSize/0x10) + 0x200;
    lBufferSizeUncpr = lBufferSizeCpr;

    CprPtr=(unsigned char*)malloc(lBufferSizeCpr + BlockSizeCompress);
    UncprPtr=(unsigned char*)malloc(lBufferSizeUncpr + BlockSizeUncompress);

    dwGetTick=GetTickCount();
    {
        z_stream zcpr;
        int ret=Z_OK;
        long lOrigToDo = lFileSize;
        long lOrigDone = 0;
        int step=0;
        memset(&zcpr,0,sizeof(z_stream));
        deflateInit(&zcpr,cprLevel);

        zcpr.next_in = FilePtr;
        zcpr.next_out = CprPtr;


        do
        {
            long all_read_before = zcpr.total_in;
            zcpr.avail_in = min(lOrigToDo,BlockSizeCompress);
            zcpr.avail_out = BlockSizeCompress;
            ret=deflate(&zcpr,(zcpr.avail_in==lOrigToDo) ? Z_FINISH : Z_SYNC_FLUSH);
            lOrigDone += (zcpr.total_in-all_read_before);
            lOrigToDo -= (zcpr.total_in-all_read_before);
            step++;
        } while (ret==Z_OK);

        lSizeCpr=zcpr.total_out;
        deflateEnd(&zcpr);
        dwGetTick=GetTickCount()-dwGetTick;
        printf("total compress size = %u, in %u step\n",lSizeCpr,step);
        printf("time = %u msec = %f sec\n\n",dwGetTick,dwGetTick/(double)1000.);
    }

    dwGetTick=GetTickCount();
    {
        z_stream zcpr;
        int ret=Z_OK;
        long lOrigToDo = lSizeCpr;
        long lOrigDone = 0;
        int step=0;
        memset(&zcpr,0,sizeof(z_stream));
        inflateInit(&zcpr);

        zcpr.next_in = CprPtr;
        zcpr.next_out = UncprPtr;


        do
        {
            long all_read_before = zcpr.total_in;
            zcpr.avail_in = min(lOrigToDo,BlockSizeUncompress);
            zcpr.avail_out = BlockSizeUncompress;
            ret=inflate(&zcpr,Z_SYNC_FLUSH);
            lOrigDone += (zcpr.total_in-all_read_before);
            lOrigToDo -= (zcpr.total_in-all_read_before);
            step++;
        } while (ret==Z_OK);

        lSizeUncpr=zcpr.total_out;
        inflateEnd(&zcpr);
        dwGetTick=GetTickCount()-dwGetTick;
        printf("total uncompress size = %u, in %u step\n",lSizeUncpr,step);
        printf("time = %u msec = %f sec\n\n",dwGetTick,dwGetTick/(double)1000.);
    }

    if (lSizeUncpr==lFileSize)
    {
        if (memcmp(FilePtr,UncprPtr,lFileSize)==0)
            printf("compare ok\n");

    }

    return 0;

}
