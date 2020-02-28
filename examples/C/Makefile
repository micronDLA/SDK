all: simpledemo threadeddemo threadedbatchdemo pollingdemo interleavingdemo twonetdemo compile quantize

compile: compile.c
	$(CC) -o $@ $< -lm -lmicrondla

simpledemo: simpledemo.c
	$(CC) -o $@ $< -lm -lmicrondla

threadeddemo: threadeddemo.c
	$(CC) -o $@ -pthread $< -lm -lmicrondla

threadedbatchdemo: threadedbatchdemo.c
	$(CC) -o $@ -pthread $< -lm -lmicrondla

pollingdemo: pollingdemo.c
	$(CC) -o $@ $< -lm -lmicrondla

interleavingdemo: interleavingdemo.c
	$(CC) -o $@ $< -lm -lmicrondla

twonetdemo: twonetdemo.c
	$(CC) -o $@ $< $(CFLAGS) -lm -lmicrondla

quantize: quantize.c
	$(CC) -g -o $@ $< -lm -lmicrondla

clean:
	rm -rf compile simpledemo threadeddemo threadedbatchdemo pollingdemo interleavingdemo twonetdemo quantize

