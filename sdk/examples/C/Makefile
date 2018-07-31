all: simpledemo \
	compile

compile: compile.c
	$(CC) -o $@ -I. $< -lm ../../libsnowflake.so

simpledemo: simpledemo.c
	$(CC) -o $@ -I. $< -lm ../../libsnowflake.so

clean:
	rm -rf compile simpledemo
