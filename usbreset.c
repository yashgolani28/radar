#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/usbdevice_fs.h>

int main(int argc, char **argv) {
    const char *filename = argv[1];
    int fd = open(filename, O_WRONLY);
    ioctl(fd, USBDEVFS_RESET, 0);
    close(fd);
    return 0;
}
