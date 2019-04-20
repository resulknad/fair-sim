import asyncio


future_list = []
def h_async(val):
    global future_list
    print(val)
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    future_list.append(future)
    if len(future_list)==5:
        future_list_bak = future_list.copy()
        future_list = []
        for ft in future_list_bak:
            ft.set_result(1)
    return future

async def opt_x():
    for i in range(10):
        h_val = h_async("some va")
        await h_val
        print(h_val.result())

async def start():
    l = []
    for i in range(5):
        l.append(asyncio.create_task(opt_x()))

    for i in l:
        await i

asyncio.run(start())
