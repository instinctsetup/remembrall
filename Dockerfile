FROM golang:1.23-alpine AS build
WORKDIR /src
COPY go.mod ./
COPY . .
RUN CGO_ENABLED=0 go build -o /out/remembrall-server ./cmd/server

FROM alpine:3.20
COPY --from=build /out/remembrall-server /usr/local/bin/remembrall-server
EXPOSE 8091
VOLUME /data
CMD ["remembrall-server", "--data", "/data/memories.json", "--addr", ":8091"]
