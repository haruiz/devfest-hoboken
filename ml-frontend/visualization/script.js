const createVisor = () => {
    tfvis.visor();
    // Create a surface on a tab
    tfvis.visor().surface({ name: 'My Surface', tab: 'My Tab' });
    // Create a surface and specify its height
    tfvis.visor().surface({
        name: 'Custom Height', tab: 'My Tab', styles: {
            height: 500
        }
    })
}

const modelSummary = () => {
    const model = tf.sequential({
        layers: [
            tf.layers.dense({ inputShape: [784], units: 32, activation: 'relu' }),
            tf.layers.dense({ units: 10, activation: 'softmax' }),
        ]
    });

    const surface = { name: 'Model Summary', tab: 'Model Inspection' };
    tfvis.show.modelSummary(surface, model);
}

const fitCallback = async () => {
    const model = tf.sequential({
        layers: [
            tf.layers.dense({ inputShape: [784], units: 32, activation: 'relu' }),
            tf.layers.dense({ units: 10, activation: 'softmax' }),
        ]
    });

    model.compile({
        optimizer: 'sgd',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    const data = tf.randomNormal([100, 784]);
    const labels = tf.randomUniform([100, 10]);

    function onBatchEnd(batch, logs) {
        console.log('Accuracy', logs.acc);
    }

    const surface = { name: 'show.fitCallbacks', tab: 'Training' };
    // Train for 5 epochs with batch size of 32.
    await model.fit(data, labels, {
        epochs: 5,
        batchSize: 32,
        callbacks: tfvis.show.fitCallbacks(surface, ['loss', 'acc']),
    });
}

const showHistory = async () => {
    const model = tf.sequential({
        layers: [
            tf.layers.dense({ inputShape: [784], units: 32, activation: 'relu' }),
            tf.layers.dense({ units: 10, activation: 'softmax' }),
        ]
    });

    model.compile({
        optimizer: 'sgd',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    const data = tf.randomNormal([100, 784]);
    const labels = tf.randomUniform([100, 10]);

    function onBatchEnd(batch, logs) {
        console.log('Accuracy', logs.acc);
    }

    const surface = { name: 'show.history', tab: 'Training' };
    // Train for 5 epochs with batch size of 32.
    const history = await model.fit(data, labels, {
        epochs: 5,
        batchSize: 32
    });

    tfvis.show.history(surface, history, ['loss', 'acc']);
}

const updateHistory = async () => {
    const model = tf.sequential({
        layers: [
            tf.layers.dense({ inputShape: [784], units: 32, activation: 'relu' }),
            tf.layers.dense({ units: 10, activation: 'softmax' }),
        ]
    });

    model.compile({
        optimizer: 'sgd',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    const data = tf.randomNormal([100, 784]);
    const labels = tf.randomUniform([100, 10]);

    function onBatchEnd(batch, logs) {
        console.log('Accuracy', logs.acc);
    }

    const surface = { name: 'show.history live', tab: 'Training' };
    // Train for 5 epochs with batch size of 32.
    const history = [];
    await model.fit(data, labels, {
        epochs: 5,
        batchSize: 32,
        callbacks: {
            onEpochEnd: (epoch, log) => {
                history.push(log);
                tfvis.show.history(surface, history, ['loss', 'acc']);
            }
        }
    });
}

const perClassAccuracy = async () => {
    const labels = tf.tensor1d([0, 0, 1, 2, 2, 2]);
    const predictions = tf.tensor1d([0, 0, 0, 2, 1, 1]);

    const result = await tfvis.metrics.perClassAccuracy(labels, predictions);
    console.log(result)

    const container = { name: 'Per Class Accuracy', tab: 'Evaluation' };
    const categories = ['cat', 'dog', 'mouse'];
    await tfvis.show.perClassAccuracy(container, result, categories);
}

const charts = () => {
    const visor = tfvis.visor();

    let data = [
        { index: 0, value: 50 },
        { index: 1, value: 100 },
        { index: 2, value: 150 },
    ];

    let surface = visor.surface({ name: 'Bar Chart', tab: 'Charts' });
    tfvis.render.barchart(surface, data);

    let rows = 5;
    let cols = 5;
    let values = [];
    for (let i = 0; i < rows; i++) {
        const row = []
        for (let j = 0; j < cols; j++) {
            row.push(Math.round(Math.random() * 50));
        }
        values.push(row);
    }
    data = { values };

    // Render to visor
    surface = visor.surface({ name: 'Confusion Matrix', tab: 'Charts' });
    //tfvis.render.confusionMatrix(surface, data);
    tfvis.render.confusionMatrix(surface, data, {
        shadeDiagonal: false
     });


    cols = 50;
    rows = 20;
    values = [];
    for (let i = 0; i < cols; i++) {
    const col = []
    for (let j = 0; j < rows; j++) {
        col.push(i * j)
    }
    values.push(col);
    }
    data = { values };

    // Render to visor
    surface = visor.surface({ name: 'Heatmap', tab: 'Charts' });
    tfvis.render.heatmap(surface, data);

   data = {
        values: [[4, 2, 8, 20], [1, 7, 2, 10], [3, 3, 20, 13]],
        xTickLabels: ['cheese', 'pig', 'font'],
        yTickLabels: ['speed', 'smoothness', 'dexterity', 'mana'],
     }
     
     // Render to visor
     surface = { name: 'Heatmap w Custom Labels', tab: 'Charts' };
     tfvis.render.heatmap(surface, data);


     const headers = [
        'Col 1',
        'Col 2',
        'Col 3',
      ];
      
      values = [
        [1, 2, 3],
        ['4', '5', '6'],
        ['<strong>7</strong>', true, false],
      ];
      
      surface = { name: 'Table', tab: 'Charts' };
      tfvis.render.table(surface, { headers, values });


      const series1 = Array(100).fill(0)
        .map(y => Math.random() * 100 - (Math.random() * 50))
        .map((y, x) => ({ x, y, }));

    const series2 = Array(100).fill(0)
        .map(y => Math.random() * 100 - (Math.random() * 150))
        .map((y, x) => ({ x, y, }));

    const series = ['First', 'Second'];
    data = { values: [series1, series2], series }

    surface = { name: 'Scatterplot', tab: 'Charts' };
    tfvis.render.scatterplot(surface, data);

    surface = { name: 'Line chart', tab: 'Charts' };
    tfvis.render.linechart(surface, data);

    data = Array(100).fill(0)
   .map(x => Math.random() * 100 - (Math.random() * 50))

    // Push some special values for the stats table.
    data.push(Infinity);
    data.push(NaN);
    data.push(0);

    surface = { name: 'Histogram', tab: 'Charts' };
    tfvis.render.histogram(surface, data);

}

window.onload = () => {
    perClassAccuracy();
}