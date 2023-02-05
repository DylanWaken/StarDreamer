import tornado.ioloop

from process_request import basicRequestHandler

if __name__ == "__main__":
    app = tornado.web.Application([(r"/", basicRequestHandler)])
    app.listen(8000)
    print("listening on port 8000")
    tornado.ioloop.IOLoop.current().start()
