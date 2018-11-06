const express = require('express');
const router = express.Router();
const _ = require('lodash');
const KerasJS = require('keras-js');

const model = new KerasJS.Model({
    filepath: appRoot + '/config/model.bin',
    filesystem: true,
    gpu: false
});
const wordDict = require(appRoot + '/config/word-dictionary.json');

router.get('/', function(req, res, next) {
    const requestParams = req.url;
    const label_ids = [0, 1]; // 0 -> normal, 1 -> anomaly
    const threshold = 0.95;
    let responseObj;

    model.ready()
      .then(() => {
        const maxInputLength = 1024;
        let logToSequence = [];
        let paddedSequence = new Float32Array(maxInputLength).fill(0);
  
        // Choosing log properties we are interested in to process
        for (let i = 0; i < requestParams.length; i++) {
          const key = requestParams[i];
          if (wordDict[key]) {
            logToSequence.push(wordDict[key]);
          }
        };
  
        // Fit log sequence to paddedSequence
        for (let i = logToSequence.length; i > -1; i--) {
          const revPos = paddedSequence.length - (logToSequence.length - i);
          paddedSequence[revPos] = logToSequence[i];
        }
  
        return model.predict({
          'input': paddedSequence
        });
      })
      .then(prediction => {
        if (_.size(prediction.output) > 0) {
          let confidence = prediction.output[0] >= threshold ? 
                                                (prediction.output[0] * 100).toFixed(2) :
                                                (100 - (prediction.output[0] * 100)).toFixed(2)
          console.log(`Malicious request confidence: ${confidence}%`);
          responseObj = {
              label: prediction.output[0] >= threshold ? label_ids[1] : label_ids[0],
              accuracy: `${confidence}%`
          };

          res.status(200).send({
              prediction: responseObj
          });
        }
      })
      .catch(err => {
        console.log(`Error: ${err}`);
        res.status(500).send({
            message: `There was an error processing the request. Please verify the logs. Error: ${err}`
        });
      })
});

module.exports = router;
