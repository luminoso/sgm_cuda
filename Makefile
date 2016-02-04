
TARGETS = lib/libcutil_x86_64.a sgm testDiffs

all: $(TARGETS)

sgm: sgm.cu lib/libcutil_x86_64.a
	nvcc -arch=sm_13 -O -Icommon/inc sgm.cu -Llib -lcutil_x86_64 -o sgm

testDiffs: testDiffs.cu lib/libcutil_x86_64.a
	nvcc -arch=sm_13 -O -Icommon/inc testDiffs.cu -Llib -lcutil_x86_64 -o testDiffs

lib/libcutil_x86_64.a: 
	make -C common

clean:
	make -C common clean
	rm -f $(TARGETS) 
	rm -f h_dbull.pgm d_dbull.pgm
