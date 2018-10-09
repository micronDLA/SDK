all: simpledemo threadeddemo threadedbatchdemo pollingdemo interleavingdemo compile

compile: compile.c
	$(CC) -o $@ $< -lm ../../libfwdnxt.so

simpledemo: simpledemo.c
	$(CC) -o $@ $< -lm ../../libfwdnxt.so

threadeddemo: threadeddemo.c
	$(CC) -o $@ -pthread $< -lm ../../libfwdnxt.so

threadedbatchdemo: threadedbatchdemo.c
	$(CC) -o $@ -pthread $< -lm ../../libfwdnxt.so

pollingdemo: pollingdemo.c
	$(CC) -o $@ $< -lm ../../libfwdnxt.so

interleavingdemo: interleavingdemo.c
	$(CC) -o $@ $< -lm ../../libfwdnxt.so

clean:
	rm -rf compile simpledemo threadeddemo threadedbatchdemo pollingdemo interleavingdemo

