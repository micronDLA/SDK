all: simpledemo threadeddemo threadedbatchdemo pollingdemo interleavingdemo compile

compile: compile.c
	$(CC) -o $@ $< -lm ../../libsnowflake.so

simpledemo: simpledemo.c
	$(CC) -o $@ $< -lm ../../libsnowflake.so

threadeddemo: threadeddemo.c
	$(CC) -o $@ -pthread $< -lm ../../libsnowflake.so

threadedbatchdemo: threadedbatchdemo.c
	$(CC) -o $@ -pthread $< -lm ../../libsnowflake.so

pollingdemo: pollingdemo.c
	$(CC) -o $@ $< -lm ../../libsnowflake.so

interleavingdemo: interleavingdemo.c
	$(CC) -o $@ $< -lm ../../libsnowflake.so

clean:
	rm -rf compile simpledemo threadeddemo threadedbatchdemo pollingdemo interleavingdemo

