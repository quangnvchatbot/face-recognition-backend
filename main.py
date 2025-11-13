const functions = require('firebase-functions');
const admin = require('firebase-admin');

admin.initializeApp();
const db = admin.database();

exports.recognizeFace = functions
  .region('asia-southeast1')
  .https.onRequest(async (req, res) => {
    
    // Enable CORS
    res.set('Access-Control-Allow-Origin', '*');
    res.set('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.set('Access-Control-Allow-Headers', 'Content-Type');
    
    if (req.method === 'OPTIONS') {
      res.status(200).send('');
      return;
    }

    try {
      // 1. Lấy dữ liệu từ request
      const { eventId } = req.body;
      
      if (!eventId) {
        return res.status(400).json({
          status: 'error',
          message: 'Missing eventId'
        });
      }

      // 2. Lấy danh sách persons từ Firebase
      const personsSnapshot = await db.ref('persons').once('value');
      const persons = personsSnapshot.val();

      if (!persons) {
        return res.status(404).json({
          status: 'error',
          message: 'No persons in database'
        });
      }

      // 3. Lấy random person (tạm thời, sau upgrade face recognition)
      const personIds = Object.keys(persons);
      const randomPerson = personIds[Math.floor(Math.random() * personIds.length)];
      const matchedPerson = persons[randomPerson];

      // 4. Check duplicate checkin
      const checkinSnapshot = await db
        .ref(`events/${eventId}/checkins`)
        .once('value');
      
      const checkins = checkinSnapshot.val() || {};
      
      // Kiểm tra xem person này đã check-in chưa
      for (const checkinId in checkins) {
        if (checkins[checkinId].person_id === randomPerson) {
          return res.status(409).json({
            status: 'duplicate',
            person_id: randomPerson,
            person_name: matchedPerson.name,
            message: 'Already checked in'
          });
        }
      }

      // 5. Tạo checkin record
      const checkinId = `checkin_${Date.now()}`;
      const checkinData = {
        person_id: randomPerson,
        person_name: matchedPerson.name,
        position: matchedPerson.position,
        department: matchedPerson.department,
        timestamp: Date.now(),
        image_url: matchedPerson.image_url
      };

      await db
        .ref(`events/${eventId}/checkins/${checkinId}`)
        .set(checkinData);

      // 6. Return success
      return res.status(200).json({
        status: 'success',
        person_id: randomPerson,
        person_name: matchedPerson.name,
        position: matchedPerson.position,
        department: matchedPerson.department,
        confidence: 0.95,
        message: 'Checkin successful'
      });

    } catch (error) {
      console.error('Error:', error);
      return res.status(500).json({
        status: 'error',
        message: error.message
      });
    }
  });
