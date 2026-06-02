let sceneID = 0;
let stride = 20;
let sourceSplatData = new Float32Array();
let splatOrder = [];

self.addEventListener("message", (event) => {
  const message = event.data;
  if (!message) return;

  try {
    switch (message.type) {
    case "setScene":
      setScene(message);
      break;
    case "sort":
      sortScene(message);
      break;
    default:
      break;
    }
  } catch (error) {
    self.postMessage({
      type: "error",
      sceneID: message.sceneID ?? sceneID,
      requestID: message.requestID ?? 0,
      message: error?.message ?? String(error),
    });
  }
});

function setScene(message) {
  sceneID = message.sceneID;
  stride = message.stride;
  sourceSplatData = new Float32Array(message.splatBuffer);
  const splatCount = sourceSplatData.length / stride;
  splatOrder = Array.from({ length: splatCount }, (_, index) => index);
  self.postMessage({
    type: "ready",
    sceneID,
    splatCount,
  });
}

function sortScene(message) {
  if (message.sceneID !== sceneID || sourceSplatData.length === 0) return;

  const start = performance.now();
  const eye = message.eye;
  const forward = message.forward;
  splatOrder.sort((lhs, rhs) => depth(rhs, eye, forward) - depth(lhs, eye, forward));
  const sortedIndices = Uint32Array.from(splatOrder);

  self.postMessage({
    type: "sorted",
    sceneID,
    requestID: message.requestID,
    cameraRevision: message.cameraRevision,
    sortDuration: performance.now() - start,
    sortedIndices: sortedIndices.buffer,
  }, [sortedIndices.buffer]);
}

function depth(index, eye, forward) {
  const offset = index * stride;
  return (
    (sourceSplatData[offset] - eye.x) * forward.x
    + (sourceSplatData[offset + 1] - eye.y) * forward.y
    + (sourceSplatData[offset + 2] - eye.z) * forward.z
  );
}
