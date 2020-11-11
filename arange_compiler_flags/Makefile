all: arange_test arange_test_avx2 arange_test_avx arange_test_avx2_fma arange_test_avx_fma

arange_test: arange_test.cpp
	g++ -o $@ $^ -std=gnu++14 -O3

arange_test_avx: arange_test.cpp
	g++ -o $@ $^ -std=gnu++14 -O3 -mavx -DCPU_CAPABILITY=AVX -DCPU_CAPABILITY_AVX -MD

arange_test_avx_fma: arange_test.cpp
	g++ -o $@ $^ -std=gnu++14 -O3 -mavx -mfma -DCPU_CAPABILITY=AVX -DCPU_CAPABILITY_AVX -MD

arange_test_avx2: arange_test.cpp
	g++ -o $@ $^ -std=gnu++14 -O3  -mavx2 -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DCPU_CAPABILITY=AVX2 -DCPU_CAPABILITY_AVX2 -MD

arange_test_avx2_fma: arange_test.cpp
	g++ -o $@ $^ -std=gnu++14 -O3  -mavx2 -mfma -mno-avx256-split-unaligned-load -mno-avx256-split-unaligned-store -DCPU_CAPABILITY=AVX2 -DCPU_CAPABILITY_AVX2 -MD


clean:
	rm -f arange_test arange_test_avx arange_test_avx_fma arange_test_avx2 arange_test_avx2_fma
	rm -f arange_test.d arange_test_avx.d arange_test_avx_fma.d arange_test_avx2.d arange_test_avx2_fma.d
